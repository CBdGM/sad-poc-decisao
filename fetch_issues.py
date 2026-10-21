"""Baixa issues fechadas e rotuladas (bug / não bug) de um repositório do GitHub.

Uso:
    python fetch_issues.py --list-labels                 # só mostra as labels do repo
    python fetch_issues.py --limit 10                    # 10 por classe
    python fetch_issues.py                               # 50 por classe (padrão)

Saída: data/issues.jsonl (não baixa de novo se o arquivo já existir; use --force).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

DATA = Path(__file__).parent / "data"
OUT = DATA / "issues.jsonl"
API = "https://api.github.com"

# Padrões para kubernetes/kubernetes (confirmados com a lista de labels do repo).
DEFAULT_BUG = "kind/bug,kind/regression"
DEFAULT_NOT_BUG = "kind/feature,kind/support,kind/documentation"
# Labels ambíguas: issues com elas ficam de fora dos dois lados.
DEFAULT_EXCLUDE = "kind/flake,kind/failing-test"


def github_token():
    tok = os.getenv("GITHUB_TOKEN")
    if tok:
        return tok
    try:  # cai para o token do gh CLI, se estiver autenticado
        return subprocess.run(["gh", "auth", "token"], capture_output=True, text=True,
                              check=True).stdout.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def session():
    s = requests.Session()
    s.headers.update({"Accept": "application/vnd.github+json",
                      "X-GitHub-Api-Version": "2022-11-28"})
    tok = github_token()
    if tok:
        s.headers["Authorization"] = f"Bearer {tok}"
    else:
        print("AVISO: sem GITHUB_TOKEN nem gh autenticado; limite de 60 req/h.", file=sys.stderr)
    return s


def get(s, url, **params):
    while True:
        r = s.get(url, params=params, timeout=30)
        if r.status_code == 403 and r.headers.get("X-RateLimit-Remaining") == "0":
            wait = max(1, int(r.headers.get("X-RateLimit-Reset", time.time() + 60)) - int(time.time()))
            print(f"Rate limit atingido, esperando {wait}s...", file=sys.stderr)
            time.sleep(wait + 1)
            continue
        r.raise_for_status()
        return r


def list_labels(s, repo):
    labels, page = [], 1
    while True:
        batch = get(s, f"{API}/repos/{repo}/labels", per_page=100, page=page).json()
        labels += [l["name"] for l in batch]
        if len(batch) < 100:
            return labels
        page += 1


def search(s, repo, labels, since, until, want):
    """Issues fechadas (não PRs) com qualquer uma das `labels`, mais recentes primeiro."""
    q = f'repo:{repo} is:issue is:closed label:{",".join(labels)} created:{since}..{until}'
    items, page = [], 1
    while len(items) < want:
        data = get(s, f"{API}/search/issues", q=q, sort="created", order="desc",
                   per_page=100, page=page).json()
        batch = [i for i in data["items"] if "pull_request" not in i]
        items += batch
        if len(data["items"]) < 100 or page * 100 >= min(data["total_count"], 1000):
            break
        page += 1
    return items


def collect(s, repo, gold, labels, conflict, since_list, want):
    """Junta issues da classe `gold` percorrendo as janelas de data em ordem até ter `want`."""
    chosen, seen, until = [], set(), "*"
    for since in since_list:
        for it in search(s, repo, labels, since, until, want * 2):
            names = {l["name"] for l in it["labels"]}
            if it["number"] in seen or names & conflict:
                continue
            seen.add(it["number"])
            chosen.append({"number": it["number"], "title": it["title"], "body": it.get("body") or "",
                           "labels": sorted(names), "created_at": it["created_at"],
                           "url": it["html_url"], "gold": gold})
            if len(chosen) >= want:
                return chosen
        print(f"  {gold}: {len(chosen)} issues desde {since}", file=sys.stderr)
        until = since  # a próxima janela termina onde esta começou
    return chosen


def dataset_path(repo: str) -> Path:
    """Arquivo de cache por repositório, usado pela interface (o CLI usa data/issues.jsonl)."""
    return DATA / "datasets" / (repo.replace("/", "__") + ".jsonl")


def fetch(repo, limit, bug, not_bug, excl, since, out, s=None):
    """Baixa `limit` issues por classe e grava em `out`. Devolve a lista de issues."""
    s = s or session()
    missing = (set(bug) | set(not_bug)) - set(list_labels(s, repo))
    if missing:
        raise ValueError(f"Labels inexistentes em {repo}: {sorted(missing)}")
    bug, not_bug, excl = set(bug), set(not_bug), set(excl)
    issues = (collect(s, repo, "bug", bug, not_bug | excl, since, limit)
              + collect(s, repo, "not_bug", not_bug, bug | excl, since, limit))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for it in issues:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    return issues


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default="kubernetes/kubernetes")
    ap.add_argument("--out", type=Path, default=OUT, help="arquivo de saída (padrão data/issues.jsonl)")
    ap.add_argument("--limit", type=int, default=50, help="issues por classe")
    ap.add_argument("--bug-labels", default=DEFAULT_BUG)
    ap.add_argument("--not-bug-labels", default=DEFAULT_NOT_BUG)
    ap.add_argument("--exclude-labels", default=DEFAULT_EXCLUDE)
    ap.add_argument("--since", default="2026-05-01,2026-03-01",
                    help="janelas de data em ordem de preferência; a seguinte só é usada se faltar issue")
    ap.add_argument("--list-labels", action="store_true", help="só lista as labels e sai")
    ap.add_argument("--force", action="store_true", help="baixa de novo mesmo se o cache existir")
    args = ap.parse_args()

    s = session()
    if args.list_labels:
        for name in list_labels(s, args.repo):
            print(name)
        return

    if args.out.exists() and not args.force:
        n = sum(1 for _ in args.out.open())
        print(f"{args.out} já existe ({n} issues). Use --force para baixar de novo.")
        return

    split = lambda v: {x.strip() for x in v.split(",") if x.strip()}
    bug, not_bug, excl = split(args.bug_labels), split(args.not_bug_labels), split(args.exclude_labels)
    since = [d.strip() for d in args.since.split(",")]
    print(f"Repo {args.repo} | bug={sorted(bug)} | não bug={sorted(not_bug)} | excluídas={sorted(excl)}")
    try:
        issues = fetch(args.repo, args.limit, bug, not_bug, excl, since, args.out, s)
    except ValueError as e:
        sys.exit(f"{e}. Rode --list-labels.")
    counts = {g: sum(i["gold"] == g for i in issues) for g in ("bug", "not_bug")}
    oldest = min(i["created_at"] for i in issues)[:10] if issues else "-"
    print(f"Salvo {args.out}: {counts} | mais antiga: {oldest}")


if __name__ == "__main__":
    main()
