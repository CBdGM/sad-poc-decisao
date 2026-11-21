"""Triagem (Sistema 1): classifica issues abertas com um modelo de decisão.

Para cada issue, UMA chamada ao modelo com três perguntas (QUESTIONS): é bug?, severidade e se há
informação para reproduzir. A revisão humana sai das respostas (human_review). Usado pela aba
Triagem do app; também roda pelo terminal:

    python triage.py --repo apache/airflow --limit 10 --days 14 --model jev
"""
import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from clean import clean_issue

ROOT = Path(__file__).parent
TRIAGES = ROOT / "results" / "triage"

QUESTIONS = {
    "is_bug": {"type": "noul",
               "instructions": "Does this issue describe a defect (bug) in the software's behavior?"},
    "severity": {"type": "score",
                 "instructions": "How severe is the problem described in this issue?",
                 "criteria": ["Low: cosmetic issue or minor inconvenience, easy workaround",
                              "Medium: a feature misbehaves for some users, a workaround exists",
                              "High: a major feature is broken or results are wrong, no easy workaround",
                              "Critical: crash, data loss, security vulnerability or outage"]},
    "repro": {"type": "noul",
              "instructions": "Does the issue give enough information to reproduce the problem "
                              "(steps, versions, error output)?"},
}
SEVERITY = ["baixa", "média", "alta", "crítica"]  # mesma ordem dos criteria

# Cortes da regra de revisão humana (ver human_review).
BUG_UNSURE = (0.2, 0.8)   # P(bug) nessa faixa = modelo indeciso
SEV_MIN = 0.5             # probabilidade do nível de severidade escolhido abaixo disso = incerta
REPRO_MIN = 0.5           # P(informação suficiente) abaixo disso = falta informação


def question_tokens(model) -> int:
    """Tokens que a maior pergunta ocupa (a de severidade, por causa dos níveis)."""
    texts = [q["instructions"] + " " + " ".join(q.get("criteria", [])) for q in QUESTIONS.values()]
    return max(model.count_tokens(t) for t in texts)


def prepare(model, issue):
    """Texto limpo + se ele cabe no modelo. A Laya corta o que passar do contexto (512 tokens)."""
    text = clean_issue(issue)
    if hasattr(model, "max_len"):
        room = model.max_len - question_tokens(model) - 8
        return text, ("truncated" if model.count_tokens(text) > room else "clean")
    return text, "clean"


def human_review(p_bug, sev_level, p_sev, p_repro, threshold=0.5):
    """Motivos para mandar a issue a um humano (lista vazia = o modelo resolve sozinho)."""
    reasons = []
    is_bug = p_bug >= threshold
    if BUG_UNSURE[0] < p_bug < BUG_UNSURE[1]:
        reasons.append("incerto se é bug")
    if p_bug >= BUG_UNSURE[0] and p_sev < SEV_MIN:
        reasons.append("severidade incerta")
    if is_bug and p_repro < REPRO_MIN:
        reasons.append("falta informação para reproduzir")
    if is_bug and sev_level == len(SEVERITY) - 1:
        reasons.append("bug crítico")
    return reasons


def decide(row, threshold=0.5):
    """Classe, severidade exibida e revisão humana a partir das respostas brutas do modelo.

    Separado da chamada ao modelo para a tela poder mudar o threshold sem perguntar de novo.
    """
    reasons = human_review(row["p_bug"], row["sev_level"], row["p_sev"], row["p_repro"], threshold)
    show_sev = row["p_bug"] >= BUG_UNSURE[0]  # severidade só faz sentido para o que pode ser bug
    return {**row, "classe": "bug" if row["p_bug"] >= threshold else "não bug",
            "severidade": SEVERITY[row["sev_level"]] if show_sev else "",
            "revisao_humana": bool(reasons), "motivos": "; ".join(reasons)}


def triage_issue(model, issue):
    """Uma chamada ao modelo com as três perguntas; devolve as respostas brutas."""
    text, source = prepare(model, issue)
    t = time.perf_counter()
    ans = model.ask(text, QUESTIONS)
    ms = (time.perf_counter() - t) * 1000
    sev = ans["severity"]
    return {"number": issue["number"], "title": issue["title"], "url": issue["url"],
            "created_at": issue["created_at"][:10], "labels": ", ".join(issue.get("labels", [])),
            "p_bug": round(ans["is_bug"]["p"], 4), "sev_level": sev["level"],
            "p_sev": round(sev["p_level"], 4), "sev_probs": [round(p, 4) for p in sev["probs"]],
            "p_repro": round(ans["repro"]["p"], 4), "input": source, "latency_ms": round(ms, 1),
            "model": model.name, "served_by": (model.last or {}).get("served_by", "")}


def sort_rows(rows):
    """Bugs primeiro, do mais grave ao mais leve; depois pela probabilidade de ser bug."""
    return sorted(rows, key=lambda r: (r["classe"] != "bug", -r["sev_level"] if r["severidade"] else 1,
                                       -r["p_bug"]))


def triage(model, issues, threshold=0.5, on_row=None):
    model.ask("warm-up", QUESTIONS)  # 1ª chamada (GPU da Laya, conexão do Jev) fora da medição
    rows = []
    for issue in issues:
        rows.append(triage_issue(model, issue))
        if on_row:
            on_row(rows)
    return sort_rows([decide(r, threshold) for r in rows])


def save(repo, params, rows, issues):
    """Grava a análise em results/triage/<data-hora>__<owner>__<repo>.json (com os textos das
    issues, para a tela mostrar original × limpo sem buscar de novo)."""
    TRIAGES.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = TRIAGES / f"{run_id}__{repo.replace('/', '__')}.json"
    path.write_text(json.dumps({"id": run_id, "repo": repo, "params": params, "rows": rows,
                                "issues": {str(i["number"]): i for i in issues}},
                               ensure_ascii=False, indent=2))
    return path


def list_saved():
    return sorted(TRIAGES.glob("*.json"), reverse=True) if TRIAGES.exists() else []


def main():
    from fetch_issues import fetch_open
    from models import JevModel, LayaModel

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--model", choices=["jev", "laya"], default="jev")
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    issues = fetch_open(args.repo, args.limit, args.days)
    model = JevModel() if args.model == "jev" else LayaModel()
    rows = triage(model, issues, args.threshold)
    path = save(args.repo, vars(args), rows, issues)
    for r in rows:
        flag = "👤" if r["revisao_humana"] else "  "
        print(f"{flag} #{r['number']:<7} {r['classe']:<8} P={r['p_bug']:.2f} {r['severidade']:<8} "
              f"{r['motivos'][:40]:<40} {r['title'][:60]}")
    print(f"\nSalvo em {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
