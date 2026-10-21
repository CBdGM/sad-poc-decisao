"""Roda os modelos de decisão sobre data/issues.jsonl e imprime as métricas.

Uso:
    python run_eval.py                          # todos os modelos disponíveis
    python run_eval.py --models laya,ollama
    python run_eval.py --models laya --summarize --limit 20
"""
import argparse
import csv
import math
import sys
import time
from pathlib import Path

from clean import clean_issue, load_issues, summarize
from models import JevModel, LayaModel, OllamaBaseline

QUESTION = "Does this issue describe a defect (bug) in the software's behavior?"
RESULTS = Path(__file__).parent / "results"


def build(name):
    if name == "laya":
        return LayaModel()
    if name == "jev":
        return JevModel()
    if name == "ollama":
        return OllamaBaseline()
    raise SystemExit(f"modelo desconhecido: {name}")


def default_models():
    names = ["laya"]
    if JevModel.available():
        names.append("jev")
    if OllamaBaseline.available():
        names.append("ollama")
    return names


def choose_state(model, issue, text, use_summary, question=QUESTION):
    """Escolhe a entrada para o modelo. Só muda algo se o texto não couber no contexto dele."""
    room = model.room_for(question) if hasattr(model, "room_for") else model.max_state_tokens
    if room is None:
        return text, "clean"
    n = model.count_tokens(text)
    if n <= room:
        return text, "clean"
    if use_summary:
        s = summarize(issue, text)
        print(f"  [{model.name}] #{issue['number']}: {n} tokens > {room}; usando resumo "
              f"({model.count_tokens(s)} tokens)", file=sys.stderr)
        return s, "summary"
    print(f"  [{model.name}] #{issue['number']}: {n} tokens > {room}; TRUNCADO "
          f"(o modelo vê só o início; use --summarize)", file=sys.stderr)
    return text, "truncated"


def predict_one(model, issue, text, threshold=0.5, summarize_long=False, question=QUESTION,
                prepared=None):
    """Faz a pergunta a um modelo para uma issue e devolve a linha do predictions.csv.

    `prepared` = (state, source) já escolhido antes (ver `evaluate`), para que gerar resumo no
    Ollama não aconteça no meio das chamadas cronometradas.
    """
    state, source = prepared or choose_state(model, issue, text, summarize_long, question)
    t = time.perf_counter()
    prob = model.ask_noul(state, question)
    ms = (time.perf_counter() - t) * 1000
    pred = "invalid" if math.isnan(prob) else ("bug" if prob >= threshold else "not_bug")
    meta = model.last or {}
    return {"number": issue["number"], "title": issue["title"], "gold": issue["gold"],
            "model": model.name, "prob": prob, "pred": pred, "latency_ms": round(ms, 1),
            "input": source, "url": issue["url"],
            # procedência (auditoria): quem respondeu segundo o servidor, e tempos internos
            "served_by": meta.get("served_by", ""), "request_id": meta.get("request_id", ""),
            "server_ms": meta.get("server_ms"), "load_ms": meta.get("load_ms"),
            "cached_tokens": meta.get("cached_tokens"), "prompt_tokens": meta.get("prompt_tokens")}


def evaluate(models, issues, threshold=0.5, summarize_long=False, question=QUESTION,
             on_row=None, should_stop=None):
    """Avalia UM MODELO DE CADA VEZ (todas as issues dele, depois o próximo).

    Alternar modelos a cada issue fazia a Laya e o Ollama disputarem a GPU do M1: na auditoria,
    a Laya sozinha deu P50 70 ms / P90 98 ms contra 137 / 763 ms intercalada. Para cada modelo,
    as entradas (inclusive resumos) são preparadas antes de começar a cronometrar.
    """
    texts = {i["number"]: clean_issue(i) for i in issues}
    rows = []
    for name, model in models.items():
        prepared = {i["number"]: choose_state(model, i, texts[i["number"]], summarize_long, question)
                    for i in issues}
        if hasattr(model, "reset_cache"):  # Ollama: zera o cache de prompt (issues já vistas)
            model.reset_cache()
        model.ask_noul("warm-up", question)  # carga do modelo / 1ª chamada lenta fora da medição
        for issue in issues:
            if should_stop and should_stop():
                return rows
            r = predict_one(model, issue, texts[issue["number"]], threshold, summarize_long,
                            question, prepared[issue["number"]])
            r["model"] = name
            rows.append(r)
            if on_row:
                on_row(r, rows)
    return rows


def balanced(issues, limit=None):
    """Alterna bug / não bug, para que um --limit mantenha as classes balanceadas."""
    bugs = [i for i in issues if i["gold"] == "bug"]
    others = [i for i in issues if i["gold"] != "bug"]
    mixed = [x for pair in zip(bugs, others) for x in pair]
    mixed += [i for i in issues if i not in mixed]
    return mixed[:limit] if limit else mixed


def percentile(values, p):
    v = sorted(values)
    if not v:
        return float("nan")
    k = (len(v) - 1) * p / 100
    lo, hi = math.floor(k), math.ceil(k)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def metrics(rows):
    tp = sum(r["gold"] == "bug" and r["pred"] == "bug" for r in rows)
    fp = sum(r["gold"] != "bug" and r["pred"] == "bug" for r in rows)
    fn = sum(r["gold"] == "bug" and r["pred"] != "bug" for r in rows)
    tn = sum(r["gold"] != "bug" and r["pred"] == "not_bug" for r in rows)
    invalid = sum(r["pred"] == "invalid" for r in rows)
    n = len(rows)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    lat = [r["latency_ms"] for r in rows]
    return {"n": n, "acc": (tp + tn) / n if n else 0.0, "prec": prec, "rec": rec, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "invalid": invalid,
            "p50": percentile(lat, 50), "p90": percentile(lat, 90)}


def print_report(by_model, models):
    names = list(by_model)
    ms = {m: metrics(by_model[m]) for m in names}
    w = 14
    print("\n" + "=" * (24 + w * len(names)))
    print(f"{'métrica':<24}" + "".join(f"{m:>{w}}" for m in names))
    print("-" * (24 + w * len(names)))
    rows = [("issues avaliadas", "n", "{:d}"), ("acurácia", "acc", "{:.3f}"),
            ("precisão (bug)", "prec", "{:.3f}"), ("recall (bug)", "rec", "{:.3f}"),
            ("F1 (bug)", "f1", "{:.3f}"), ("latência P50 (ms)", "p50", "{:.0f}"),
            ("latência P90 (ms)", "p90", "{:.0f}")]
    for label, key, fmt in rows:
        print(f"{label:<24}" + "".join(f"{fmt.format(ms[m][key]):>{w}}" for m in names))
    for m in names:
        if isinstance(models[m], OllamaBaseline):
            o = models[m]
            rate = o.invalid / o.calls if o.calls else 0.0
            print(f"{'JSON inválido (' + m + ')':<24}{o.invalid}/{o.calls} = {rate:.1%}")

    for m in names:
        x = ms[m]
        print(f"\nMatriz de confusão — {m}   (linhas = rótulo real, colunas = previsão)")
        print(f"{'':>12}{'bug':>10}{'não bug':>10}" + (f"{'inválido':>10}" if x["invalid"] else ""))
        inv_bug = sum(r["gold"] == "bug" and r["pred"] == "invalid" for r in by_model[m])
        inv_nb = x["invalid"] - inv_bug
        print(f"{'bug':>12}{x['tp']:>10}{x['fn'] - inv_bug:>10}" + (f"{inv_bug:>10}" if x["invalid"] else ""))
        print(f"{'não bug':>12}{x['fp']:>10}{x['tn']:>10}" + (f"{inv_nb:>10}" if x["invalid"] else ""))

    for m in names:
        wrong = [r for r in by_model[m] if r["pred"] in ("bug", "not_bug") and r["pred"] != r["gold"]]
        # confiança na resposta errada: prob se previu bug, 1 - prob se previu não bug
        wrong.sort(key=lambda r: r["prob"] if r["pred"] == "bug" else 1 - r["prob"], reverse=True)
        print(f"\n5 piores erros — {m}")
        if not wrong:
            print("  (nenhum erro)")
        for r in wrong[:5]:
            conf = r["prob"] if r["pred"] == "bug" else 1 - r["prob"]
            print(f"  #{r['number']:<7} real={r['gold']:<8} prev={r['pred']:<8} conf={conf:.2f}  "
                  f"{r['title'][:70]}\n           {r['url']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", help="lista separada por vírgula: laya,jev,ollama "
                                     "(padrão: os disponíveis)")
    ap.add_argument("--summarize", action="store_true",
                    help="usa o resumo do Ollama quando o texto não cabe no contexto do modelo")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--limit", type=int, help="avalia só as N primeiras issues (alternando classes)")
    ap.add_argument("--data", type=Path, default=Path(__file__).parent / "data" / "issues.jsonl")
    args = ap.parse_args()

    issues = load_issues(args.data)
    if args.limit:
        issues = balanced(issues, args.limit)

    names = [n.strip() for n in args.models.split(",")] if args.models else default_models()
    if "ollama" in names and not OllamaBaseline.available():
        raise SystemExit("Ollama não está respondendo em localhost:11434. Abra o app ou rode `ollama serve`.")
    if args.summarize and not OllamaBaseline.available():
        raise SystemExit("--summarize precisa do Ollama ativo.")
    print(f"Modelos: {names} | issues: {len(issues)} | threshold: {args.threshold}", file=sys.stderr)

    models = {n: build(n) for n in names}  # cada modelo é carregado uma vez só
    for m in models.values():  # aquecimento (a 1ª chamada da Laya no MPS é lenta)
        if isinstance(m, LayaModel):
            m.ask_noul("warm-up", QUESTION)

    total = len(issues) * len(models)

    def progress(r, rows):
        if len(rows) % 10 == 0 or len(rows) == total:
            print(f"  {len(rows)}/{total} chamadas ({r['model']})", file=sys.stderr)

    by_model = {n: [] for n in names}
    for r in evaluate(models, issues, args.threshold, args.summarize, on_row=progress):
        by_model[r["model"]].append(r)

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / "predictions.csv"
    with out.open("w", newline="") as f:
        cols = ["number", "title", "gold", "model", "prob", "pred", "latency_ms", "input", "url",
                "served_by", "request_id", "server_ms", "load_ms",
                "cached_tokens", "prompt_tokens"]
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for rows in by_model.values():
            w.writerows(rows)
    print(f"Predições salvas em {out}", file=sys.stderr)

    print_report(by_model, models)


if __name__ == "__main__":
    main()
