"""Interface única DecisionModel e as implementações Laya, Jev e baseline Ollama."""
import json
import os
import sys
import time

import requests
from dotenv import load_dotenv

load_dotenv()

# O padrão do Ollama é 4.096 tokens, pouco para issues com logs longos (dá HTTP 500).
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "16384"))


class DecisionModel:
    name: str = "base"
    # Tokens de `state` que cabem por pergunta; None = sem limite prático.
    max_state_tokens = None
    # Procedência da última chamada, para auditoria: quem de fato respondeu (segundo o servidor),
    # id da requisição, tempo gasto dentro do servidor e tempo de carga do modelo (Ollama).
    last: dict = {}

    def ask_noul(self, state: str, question: str) -> float:
        """Probabilidade (0–1) de a resposta ser "sim"."""
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        return len(text) // 4  # estimativa grosseira; a Laya sobrescreve com o tokenizer real


# ---------------------------------------------------------------- Laya (local)

class LayaModel(DecisionModel):
    """Laya, checkpoint em inglês (ModernBERT-large, contexto de 512 tokens).

    API descoberta no README do pacote e em laya/agent.py (v0.4.12):
        agent = laya.load("convaiinnovations/laya")
        agent.predict(state, {"id": {"type": "noul", "instructions": "..."}})
        -> {"answers": {"id": {"type": "noul", "noul": 0.85, ...}}, ...}
    O texto que passar do contexto é cortado pela própria Laya (mantém o início).
    """
    name = "laya"
    MODEL_ID = "convaiinnovations/laya"  # repo raiz = checkpoint em inglês

    def __init__(self, device=None):
        import laya  # import tardio: torch só é carregado se a Laya for usada
        t = time.time()
        self.agent = laya.load(self.MODEL_ID, device=device)
        self.max_len = int(self.agent.cfg.get("max_len", 512))
        sub = f"/{self.agent.subfolder}" if self.agent.subfolder else ""
        rev = (getattr(self.agent, "revision", None) or "?")[:12]
        # checkpoint e revisão (commit no Hugging Face) que foram realmente carregados
        self.served_by = f"{self.agent.model_id}{sub}@{rev}"
        print(f"[laya] carregada em {time.time() - t:.1f}s ({self.served_by}, "
              f"device={self.agent.device}, max_len={self.max_len})", file=sys.stderr)

    def count_tokens(self, text: str) -> int:
        return len(self.agent.tok(text, add_special_tokens=False)["input_ids"])

    def room_for(self, question: str) -> int:
        # contexto - pergunta - marcadores especiais (margem de 8 tokens)
        return self.max_len - self.count_tokens(question) - 8

    def ask_noul(self, state: str, question: str) -> float:
        r = self.agent.predict(state, {"q": {"type": "noul", "instructions": question}})
        self.last = {"served_by": self.served_by}
        return float(r["answers"]["q"]["noul"])


# ---------------------------------------------------------------- Jev (TypeSafe, nuvem)

def parse_jev_noul(response: dict, qid: str) -> float:
    """Extrai a probabilidade de "sim" da resposta do /v1/systemone.

    Formato em https://docs.typesafe.ai/api.md (lido em 2026-10-19):
        {"model": "jev-1.13.0",
         "answers": {"<qid>": {"type": "noul", "noul": 0.95}},
         "usage": {"input_tokens": 296, "output_tokens": 20}}
    Conferido numa chamada real em 20/10/2026 (jev-1.13.0): a resposta veio exatamente nesse formato.
    """
    ans = response["answers"][qid]
    if ans.get("type") != "noul":
        raise ValueError(f"resposta inesperada do Jev: {ans}")
    return float(ans["noul"])


class JevModel(DecisionModel):
    name = "jev"
    max_state_tokens = 30_000  # 32k por pergunta (state + pergunta), com folga

    def __init__(self):
        self.key = os.getenv("TYPESAFE_API_KEY")
        if not self.key:
            raise RuntimeError("TYPESAFE_API_KEY não definida")
        self.base = os.getenv("TYPESAFE_BASE_URL", "https://api.typesafe.ai").rstrip("/")
        self.model = os.getenv("TYPESAFE_MODEL", "jev-latest")
        self.s = requests.Session()
        self.s.headers.update({"Authorization": f"Bearer {self.key}",
                               "Content-Type": "application/json"})

    @staticmethod
    def available() -> bool:
        return bool(os.getenv("TYPESAFE_API_KEY"))

    def ask_noul(self, state: str, question: str) -> float:
        body = {"model": self.model, "state": state,
                "questions": {"is_bug": {"type": "noul", "instructions": question}}}
        for attempt in range(5):
            r = self.s.post(f"{self.base}/v1/systemone", json=body, timeout=60)
            if r.status_code in (429, 529):  # rate limit / sobrecarga: backoff
                time.sleep(float(r.headers.get("retry-after", 2 ** attempt)))
                continue
            if r.status_code >= 400:
                raise RuntimeError(f"Jev HTTP {r.status_code}: {r.text[:300]}")
            data = r.json()
            upstream = r.headers.get("x-envoy-upstream-service-time")
            self.last = {"served_by": data.get("model", "?"),  # versão que respondeu, ex. jev-1.13.0
                         "request_id": r.headers.get("x-typesafe-request-id", ""),
                         "server_ms": float(upstream) if upstream else None,
                         "retries": attempt}
            return parse_jev_noul(data, "is_bug")
        raise RuntimeError("Jev: muitas tentativas com 429/529")


# ---------------------------------------------------------------- baseline: só LLM (Ollama)

BASELINE_PROMPT = """You are triaging a GitHub issue.
Question: {question}
Answer only with a JSON object of the form {{"is_bug": true}} or {{"is_bug": false}}.

ISSUE:
{state}"""


class OllamaBaseline(DecisionModel):
    """O mesmo LLM do Ollama respondendo direto. Devolve 1.0/0.0, ou NaN se o JSON vier inválido."""
    name = "ollama"

    def __init__(self, model=None, url=None):
        self.model = model or os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
        self.url = (url or os.getenv("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        self.calls = 0
        self.invalid = 0
        self.invalid_samples = []

    @staticmethod
    def available(url=None) -> bool:
        url = (url or os.getenv("OLLAMA_URL", "http://localhost:11434")).rstrip("/")
        try:
            return requests.get(f"{url}/api/tags", timeout=2).ok
        except requests.RequestException:
            return False

    def reset_cache(self):
        """Descarrega o modelo (zera o cache de prompt) e recarrega com uma chamada de aquecimento.

        Sem isso, avaliar de novo as mesmas issues mede respostas vindas do cache e o Ollama parece
        muito mais rápido do que seria com issues novas.
        """
        requests.post(f"{self.url}/api/generate", json={"model": self.model, "keep_alive": 0}, timeout=60)
        for _ in range(50):  # espera o runner sair da memória
            loaded = requests.get(f"{self.url}/api/ps", timeout=5).json().get("models", [])
            if not any(m["name"] == self.model for m in loaded):
                break
            time.sleep(0.2)

    def ask_noul(self, state: str, question: str) -> float:
        self.calls += 1
        self.last = {"served_by": ""}
        try:
            r = requests.post(f"{self.url}/api/chat", timeout=300, json={
                "model": self.model, "stream": False, "format": "json",
                # keep_alive: sem isso o Ollama descarrega o modelo após 5 min parado e a próxima
                # chamada mede também a recarga (~4 s), inflando a latência.
                "keep_alive": "30m",
                "options": {"temperature": 0, "num_ctx": OLLAMA_NUM_CTX},
                "messages": [{"role": "user",
                              "content": BASELINE_PROMPT.format(question=question, state=state)}]})
            r.raise_for_status()
        except requests.RequestException as e:  # uma falha isolada conta como resposta inválida
            print(f"  [ollama] erro HTTP: {e}", file=sys.stderr)
            self.invalid += 1
            self.invalid_samples.append(f"HTTP: {e}")
            return float("nan")
        data = r.json()
        self.last = {"served_by": data.get("model", "?"),
                     "server_ms": data.get("total_duration", 0) / 1e6,  # tempo dentro do Ollama
                     "load_ms": data.get("load_duration", 0) / 1e6,     # parte dele gasta carregando
                     # tokens do prompt reaproveitados do cache: se a issue já foi vista, a
                     # chamada fica 3-13x mais rápida que numa issue nova (medido na auditoria)
                     "cached_tokens": data.get("prompt_eval_cached_count", 0),
                     "prompt_tokens": data.get("prompt_eval_count", 0)}
        raw = data["message"]["content"]
        try:
            value = json.loads(raw)["is_bug"]
            if not isinstance(value, bool):
                raise TypeError(value)
            return 1.0 if value else 0.0
        except (ValueError, KeyError, TypeError):
            self.invalid += 1
            self.invalid_samples.append(raw[:200])
            return float("nan")
