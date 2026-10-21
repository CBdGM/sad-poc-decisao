"""Limpeza da entrada: camada 1 (regex, sempre) e camada 2 (resumo via Ollama, opcional).

Uso:
    python clean.py --show 2              # mostra antes/depois de 2 issues (1 bug, 1 não bug)
    python clean.py --show 2 --summarize  # idem, com o resumo do Ollama
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

DATA = Path(__file__).parent / "data"
SUMMARIES = DATA / "summaries.jsonl"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "16384"))  # padrão do Ollama (4.096) é pouco
CODE_KEEP_LINES = 10

# ---------------------------------------------------------------- camada 1: regex

HTML_COMMENT = re.compile(r"<!--.*?(-->|$)", re.S)
HTML_TAGS = re.compile(r"</?(details|summary|br|p|sub|sup|b|i|kbd)\b[^>]*>", re.I)
# Comandos do Prow/bots que entregam o rótulo (/kind bug, /sig node, /priority ...).
BOT_LINE = re.compile(
    r"^[ \t>]*/(kind|priority|sig|triage|area|wg|remove-[\w-]+|label|assign|unassign|cc|uncc|"
    r"lifecycle|milestone|close|reopen|retitle|hold|good-first-issue|help|ok-to-test|lgtm|approve)"
    r"\b.*$", re.M | re.I)
BOT_INLINE = re.compile(r"(?<!\S)/(kind|priority|sig|triage|area|wg)\s+[\w/-]+", re.I)
# Cabeçalhos de template (bug, feature, support, docs). Também entregam o rótulo.
TEMPLATE_HEADING = re.compile(
    r"^#{1,6}\s*(what happened\??|what did you expect to happen\??|"
    r"how can we reproduce it.*|anything else we need to know\??|kubernetes version|"
    r"cloud provider|os version|install tools|container runtime.*|related plugins.*|"
    r"what would you like to be added\??|why is this needed\??|"
    r"which jobs are (failing|flaking).*|which tests? (are|is) (failing|flaking).*|"
    r"since when has it been (failing|flaking).*|testgrid link|reason for failure.*|"
    r"(bug|feature|enhancement)( request| report)?|describe the (bug|feature).*|"
    r"expected behavio(u)?r|actual behavio(u)?r|steps to reproduce)\s*:?\s*$", re.M | re.I)
# Seções de formulário que existem só num dos templates (em geral, ambiente/versões no de bug).
# Saem inteiras, com título e conteúdo: numa auditoria do Kubernetes, "kubectl version" aparecia em
# 46% dos bugs e 2% dos não bugs; no Airflow, versão/Deployment/Helm só existem no template de bug.
TEMPLATE_DROP_SECTION = re.compile(
    r"^#{1,6}\s*("
    # Kubernetes
    r"kubernetes version|cloud provider|os version|install tools|container runtime[^\n]*|"
    r"related plugins[^\n]*|"
    # Apache Airflow
    r"under which category[^\n]*|apache airflow version|operating system|deployment( details)?|"
    r"apache airflow provider[^\n]*|versions of apache airflow providers|official helm chart version|"
    r"helm chart configuration|docker image customizations|are you willing to submit[^\n]*|"
    r"code of conduct|related issues"
    r")\s*:?\s*$.*?(?=^#{1,6}\s|\Z)", re.M | re.S | re.I)
# Qualquer outro cabeçalho Markdown sai inteiro (só o conteúdo abaixo fica): títulos de template
# como "Use case/motivation" ou "What happened?" diriam a classe mesmo sem o "#".
OTHER_HEADING = re.compile(r"^#{1,6}[ \t]+.*$", re.M)
CHECKBOX = re.compile(r"^\s*[-*]\s*\[[ xX]\].*$", re.M)
NO_RESPONSE = re.compile(r"^\s*_No response_\s*$", re.M)
IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)|<img\b[^>]*>", re.I)
CODE_BLOCK = re.compile(r"(```|~~~)[^\n]*\n(.*?)(\n\1|\Z)", re.S)
TITLE_PREFIX = re.compile(
    r"^\s*[\[(]?\s*(bug|feature( request)?|enhancement|question|support|docs?|"
    r"documentation|regression|failing test|flaky test|flake)\s*[\])]?\s*[:\-]?\s+", re.I)


def _shorten_code(m):
    lines = m.group(2).split("\n")
    if len(lines) <= CODE_KEEP_LINES:
        return "```\n" + m.group(2) + "\n```"
    kept = "\n".join(lines[:CODE_KEEP_LINES])
    return f"```\n{kept}\n[... {len(lines) - CODE_KEEP_LINES} more lines]\n```"


def clean_body(body: str) -> str:
    t = (body or "").replace("\r\n", "\n")
    t = HTML_COMMENT.sub("", t)
    # Tira os blocos de código do caminho (placeholders) para que as regras abaixo
    # não mexam no conteúdo deles (ex.: um "# comentário" não é cabeçalho).
    blocks = []

    def stash(m):
        blocks.append(_shorten_code(m))
        return f"\x00{len(blocks) - 1}\x00"

    t = CODE_BLOCK.sub(stash, t)
    t = BOT_LINE.sub("", t)
    t = BOT_INLINE.sub("", t)
    t = TEMPLATE_DROP_SECTION.sub("", t)  # depois do stash: um "# comentário" no código não corta a seção
    t = TEMPLATE_HEADING.sub("", t)
    t = OTHER_HEADING.sub("", t)
    t = CHECKBOX.sub("", t)
    t = NO_RESPONSE.sub("", t)
    t = HTML_TAGS.sub("", t)
    t = IMAGE.sub("[image]", t)
    t = re.sub(r"[ \t]+\n", "\n", t)
    t = re.sub(r"(?<=\S)[ \t]{2,}", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    t = re.sub(r"\x00(\d+)\x00", lambda m: blocks[int(m.group(1))], t)
    return t.strip()


def clean_title(title: str) -> str:
    return TITLE_PREFIX.sub("", title or "").strip()


def clean_issue(issue: dict) -> str:
    """Texto limpo: título + linha em branco + corpo limpo."""
    body = clean_body(issue.get("body", ""))
    title = clean_title(issue.get("title", ""))
    return f"{title}\n\n{body}" if body else title


# ---------------------------------------------------------------- camada 2: resumo via Ollama

SUMMARY_PROMPT = """You condense GitHub issues for a downstream classifier.
Rewrite the issue below in at most 120 words, keeping only facts:
- what the user did or wants to do,
- what they expected,
- what actually happened (errors, outputs),
- relevant environment (versions, platform).
Rules:
- Only state what the issue text says. Do not infer, guess or add details. If the expected
  behavior is not stated, do not mention it.
- Do NOT classify the issue and do NOT say whether it is a bug, a feature, a question or a request.
- Never use the words "bug", "defect" or "feature request".
- No opinions, no recommendations, no preamble. Plain prose, no headings, no lists.

ISSUE:
{text}"""


def _load_cache():
    cache = {}
    if SUMMARIES.exists():
        for line in SUMMARIES.open():
            r = json.loads(line)
            cache[r["key"]] = r["summary"]
    return cache


_cache = None


def summarize(issue: dict, text: str = None, model: str = OLLAMA_MODEL) -> str:
    """Resumo factual (≤ ~120 palavras) via Ollama, com cache em data/summaries.jsonl."""
    global _cache
    if _cache is None:
        _cache = _load_cache()
    text = text if text is not None else clean_issue(issue)
    digest = hashlib.sha1((SUMMARY_PROMPT + text).encode()).hexdigest()[:12]  # muda se o prompt mudar
    key = f"{issue['number']}:{model}:{digest}"
    if key in _cache:
        return _cache[key]
    r = requests.post(f"{OLLAMA_URL}/api/chat", timeout=300, json={
        "model": model, "stream": False, "keep_alive": "30m", "options": {"temperature": 0, "num_ctx": OLLAMA_NUM_CTX},
        "messages": [{"role": "user", "content": SUMMARY_PROMPT.format(text=text)}]})
    r.raise_for_status()
    summary = r.json()["message"]["content"].strip()
    DATA.mkdir(exist_ok=True)
    with SUMMARIES.open("a") as f:
        f.write(json.dumps({"key": key, "number": issue["number"], "summary": summary},
                           ensure_ascii=False) + "\n")
    _cache[key] = summary
    return summary


def load_issues(path=DATA / "issues.jsonl"):
    return [json.loads(l) for l in open(path)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show", type=int, default=2, help="quantas issues mostrar")
    ap.add_argument("--number", type=int, action="append", help="issue específica (pode repetir)")
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()

    issues = load_issues()
    if args.number:
        picked = [i for i in issues if i["number"] in args.number]
    else:  # alterna bug / não bug
        bugs = [i for i in issues if i["gold"] == "bug"]
        others = [i for i in issues if i["gold"] != "bug"]
        picked = [x for pair in zip(bugs, others) for x in pair][:args.show]

    for it in picked:
        raw = f"{it['title']}\n\n{it['body']}"
        cleaned = clean_issue(it)
        print("=" * 100)
        print(f"#{it['number']} [{it['gold']}]  {len(raw)} -> {len(cleaned)} caracteres")
        print("-" * 40, "ANTES", "-" * 40)
        print(raw)
        print("-" * 40, "DEPOIS", "-" * 40)
        print(cleaned)
        if args.summarize:
            print("-" * 40, f"RESUMO ({OLLAMA_MODEL})", "-" * 40)
            print(summarize(it, cleaned))


if __name__ == "__main__":
    main()
