"""Auditoria: confere se as respostas vêm do modelo certo e se as métricas e os tempos estão corretos.

    python audit.py                       # checagens offline em todas as execuções salvas
    python audit.py --live RUN_ID         # + reprodução das respostas e benchmark de latência
    python audit.py --live RUN_ID --n 10  # usa 10 issues da execução (padrão: 6)

Offline (rápido, não chama modelo nenhum), para cada execução em results/runs/:
  1. estrutura: cada modelo tem uma linha por issue, sem duplicatas, e as mesmas issues;
  2. gabarito: o `gold` de cada linha bate com o arquivo de issues baixado;
  3. previsão: `pred` é exatamente prob >= threshold (e "invalid" só quando prob é NaN);
  4. métricas: acurácia/precisão/recall/F1 recalculados com scikit-learn e percentis com numpy;
  5. procedência: quem respondeu (segundo o servidor) é o modelo pedido;
  6. latência: avisa se a execução rodou ao mesmo tempo que outra (disputa de GPU/rede).
Ao vivo (--live): pergunta de novo aos modelos e compara com o que está salvo; prova que a Laya
não acessa a rede; e mede a latência de cada modelo sozinho, sem disputa.
"""
import argparse
import json
import math
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from run_eval import metrics

ROOT = Path(__file__).parent
RUNS = ROOT / "results" / "runs"
OK, WARN, FAIL = "✅", "⚠️ ", "❌"
problems = {"fail": 0, "warn": 0}


def say(level, msg):
    if level == FAIL:
        problems["fail"] += 1
    elif level == WARN:
        problems["warn"] += 1
    print(f"  {level} {msg}")


def load_run(run_dir):
    cfg = json.loads((run_dir / "config.json").read_text())
    df = pd.read_csv(run_dir / "predictions.csv")
    df["prob"] = pd.to_numeric(df["prob"], errors="coerce")
    df["number"] = df["number"].astype(int)
    return cfg, df


def run_window(run_id, cfg, df):
    """Início e fim aproximados da execução (para detectar sobreposição entre execuções)."""
    start = datetime.strptime(run_id, "%Y%m%d-%H%M%S")
    if cfg.get("finished_at"):
        end = datetime.strptime(cfg["finished_at"], "%Y%m%d-%H%M%S")
    else:  # execuções do terminal: estima pela soma das latências
        end = start + pd.Timedelta(milliseconds=float(df["latency_ms"].sum()))
    return start, end


def audit_offline(run_dir, windows):
    cfg, df = load_run(run_dir)
    print(f"\n━━ {run_dir.name} · {cfg['repo']} · {','.join(cfg['models'])} · "
          f"{cfg['issues']} issues · status {cfg.get('status')}")
    complete = cfg.get("status") == "done"

    # 1. estrutura
    models = list(dict.fromkeys(df["model"]))
    if set(models) != set(cfg["models"]):
        say(FAIL, f"modelos no CSV {models} ≠ modelos pedidos {cfg['models']}")
    sets = {m: set(df[df.model == m].number) for m in models}
    for m in models:
        n, dup = (df.model == m).sum(), df[df.model == m].number.duplicated().sum()
        if dup:
            say(FAIL, f"{m}: {dup} issues repetidas")
        if complete and n != cfg["issues"]:
            say(FAIL, f"{m}: {n} linhas, esperado {cfg['issues']}")
    if len({frozenset(s) for s in sets.values()}) > 1 and complete:
        say(FAIL, "os modelos não avaliaram exatamente as mesmas issues")
    if not complete:
        say(WARN, f"execução incompleta ({cfg.get('status')}): métricas valem só para o que rodou")

    # 2. gabarito
    ds = ROOT / cfg.get("dataset", "")
    if ds.is_file():
        gold = {json.loads(l)["number"]: json.loads(l)["gold"] for l in ds.open()}
        wrong = [n for n, g in zip(df.number, df.gold) if gold.get(n) != g]
        say(FAIL if wrong else OK, f"gabarito confere com {ds.relative_to(ROOT)}"
            + (f" — divergem: {sorted(set(wrong))[:5]}" if wrong else ""))
    else:
        say(WARN, f"arquivo de issues {cfg.get('dataset')} não encontrado; gabarito não conferido")

    # 3. previsão = prob >= threshold
    thr = float(cfg["threshold"])
    expected = np.where(df.prob.isna(), "invalid", np.where(df.prob >= thr, "bug", "not_bug"))
    bad = (df.pred.values != expected).sum()
    say(FAIL if bad else OK, f"previsão = prob ≥ {thr} em todas as linhas" + (f" ({bad} divergem)" if bad else ""))
    if "ollama" in models:
        vals = set(df[df.model == "ollama"].prob.dropna().unique())
        say(OK if vals <= {0.0, 1.0} else FAIL, f"ollama só devolve 0/1 (visto: {sorted(vals)})")

    # 4. métricas e percentis, recalculados com bibliotecas independentes
    for m in models:
        g = df[df.model == m]
        mine = metrics(g.to_dict("records"))
        yt, yp = g.gold.values, g.pred.values  # "invalid" conta como erro, como na PoC
        ref = {"acc": accuracy_score(yt, yp),
               "prec": precision_score(yt, yp, labels=["bug"], average="macro", zero_division=0),
               "rec": recall_score(yt, yp, labels=["bug"], average="macro", zero_division=0),
               "f1": f1_score(yt, yp, labels=["bug"], average="macro", zero_division=0),
               "p50": float(np.percentile(g.latency_ms, 50)), "p90": float(np.percentile(g.latency_ms, 90))}
        diffs = {k: (mine[k], v) for k, v in ref.items() if not math.isclose(mine[k], v, abs_tol=1e-9)}
        say(FAIL if diffs else OK, f"{m}: acc {mine['acc']:.3f} · prec {mine['prec']:.3f} · rec {mine['rec']:.3f} · "
            f"F1 {mine['f1']:.3f} · P50 {mine['p50']:.0f} ms · P90 {mine['p90']:.0f} ms = scikit-learn/numpy"
            + (f" — DIVERGEM: {diffs}" if diffs else ""))

    # 5. procedência
    if "served_by" not in df.columns:
        say(WARN, "sem coluna de procedência (execução anterior a 21/10); quem respondeu não foi registrado")
    else:
        for m in models:
            served = set(df[df.model == m].served_by.dropna().astype(str)) - {""}
            if m == "laya":
                ok = served and all(s.startswith("convaiinnovations/laya@") for s in served)
            elif m == "jev":
                ok = served and all(s.startswith("jev-") for s in served)
            else:
                ok = served == {cfg.get("ollama_model", "qwen2.5:7b")}
            say(OK if ok else FAIL, f"{m}: respondido por {sorted(served) or '—'}")
            if m == "jev":
                ids = df[df.model == m].request_id.dropna()
                say(OK if ids.is_unique and len(ids) == (df.model == m).sum() else FAIL,
                    f"jev: {ids.nunique()} request-ids únicos da TypeSafe para {(df.model == m).sum()} chamadas")

    # 6. latência: sobreposição com outras execuções
    me = windows[run_dir.name]
    overl = [r for r, (s, e) in windows.items() if r != run_dir.name and s < me[1] and me[0] < e]
    if overl:
        say(WARN, f"rodou ao mesmo tempo que {overl}: latências de Laya/Ollama podem estar infladas "
                  "(disputa da GPU); compare com o benchmark do --live")
    if cfg.get("summarize"):
        say(OK, "a latência não inclui gerar o resumo (é tempo só do modelo de decisão)")
    if "cached_tokens" in df.columns and (df.model == "ollama").any():
        o = df[df.model == "ollama"]
        frac = o.cached_tokens.fillna(0).sum() / max(o.prompt_tokens.fillna(0).sum(), 1)
        say(WARN if frac > 0.5 else OK, f"ollama: {frac:.0%} dos tokens de prompt vieram do cache"
            + (" — latência otimista (issues já vistas); rode de novo com o cache zerado" if frac > 0.5 else ""))
    if "load_ms" in df.columns and (df.load_ms.fillna(0) > 500).any():
        n = int((df.load_ms.fillna(0) > 500).sum())
        say(WARN, f"ollama: {n} chamadas incluíram recarregar o modelo na memória (load > 0,5 s)")
    return cfg, df


# ================================================================ ao vivo

def live(run_id, n):
    from clean import clean_issue, load_issues, summarize
    from models import JevModel, LayaModel, OllamaBaseline
    from run_eval import QUESTION, balanced

    cfg, df = load_run(RUNS / run_id)
    issues = {i["number"]: i for i in load_issues(ROOT / cfg["dataset"])}
    picked = balanced([issues[x] for x in dict.fromkeys(df.number)], n)
    question, models = cfg["question"], {}
    for name in cfg["models"]:
        models[name] = (LayaModel() if name == "laya" else JevModel() if name == "jev"
                        else OllamaBaseline(model=cfg.get("ollama_model")))
    print(f"\n━━ AO VIVO · {run_id} · {len(picked)} issues · cada modelo sozinho, um de cada vez")

    def state_for(m, issue, source):
        text = clean_issue(issue)
        return summarize(issue, text) if source == "summary" else text

    for name, m in models.items():
        m.ask_noul("warm-up", question)  # aquecimento fora da medição
        rows, lat, srv = [], [], []
        blocked = 0
        for issue in picked:
            saved = df[(df.model == name) & (df.number == issue["number"])].iloc[0]
            state = state_for(m, issue, saved["input"])
            if name == "laya":  # prova de que a Laya não usa a rede: qualquer conexão falha
                orig = socket.socket.connect

                def no_net(*a, **k):
                    nonlocal blocked
                    blocked += 1
                    raise OSError("rede bloqueada pela auditoria")
                socket.socket.connect = no_net
            t = time.perf_counter()
            try:
                p = m.ask_noul(state, question)
            finally:
                if name == "laya":
                    socket.socket.connect = orig
            lat.append((time.perf_counter() - t) * 1000)
            if m.last.get("server_ms") is not None:
                srv.append(m.last["server_ms"])
            rows.append((issue["number"], saved["prob"], p, m.last.get("served_by")))
        diffs = [abs(a - b) for _, a, b, _ in rows if not (math.isnan(a) and math.isnan(b))]
        same_pred = sum((a >= cfg["threshold"]) == (b >= cfg["threshold"]) for _, a, b, _ in rows)
        print(f"\n  [{name}] respondido por: {sorted({r[3] for r in rows})}")
        tol = 1e-3
        level = OK if max(diffs, default=0) <= tol else (WARN if same_pred == len(rows) else FAIL)
        say(level, f"{name}: reprodução — diferença máx. de probabilidade {max(diffs, default=0):.4f}; "
                   f"mesma previsão em {same_pred}/{len(rows)}")
        if name == "laya":
            say(OK if blocked == 0 else FAIL, f"laya: {blocked} tentativas de conexão de rede durante as previsões")
        p50, p90 = np.percentile(lat, 50), np.percentile(lat, 90)
        saved_lat = df[(df.model == name) & df.number.isin([r[0] for r in rows])].latency_ms
        line = (f"{name}: latência sozinho P50 {p50:.0f} ms · P90 {p90:.0f} ms "
                f"(na execução salva: P50 {np.percentile(saved_lat, 50):.0f} · P90 {np.percentile(saved_lat, 90):.0f})")
        if srv:
            line += f" · tempo dentro do servidor P50 {np.percentile(srv, 50):.0f} ms"
        say(OK, line)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--live", metavar="RUN_ID", help="reproduz respostas e mede latência desta execução")
    ap.add_argument("--n", type=int, default=6, help="issues usadas no modo --live")
    args = ap.parse_args()

    dirs = [d for d in sorted(RUNS.glob("*")) if (d / "config.json").exists() and (d / "predictions.csv").exists()]
    windows = {}
    for d in dirs:
        cfg, df = load_run(d)
        windows[d.name] = run_window(d.name, cfg, df)
    for d in dirs:
        audit_offline(d, windows)
    if args.live:
        live(args.live, args.n)
    print(f"\nResumo: {problems['fail']} falhas, {problems['warn']} avisos")
    sys.exit(1 if problems["fail"] else 0)


if __name__ == "__main__":
    main()
