"""Interface gráfica da PoC (Streamlit).

    .venv/bin/streamlit run app.py

Configure na barra lateral (repositório, labels, issues, modelos, pergunta) e acompanhe a execução,
os resultados e o histórico nas abas.
"""
import json
import math
import re
import shutil
import threading
import traceback
from datetime import datetime
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import fetch_issues as fi
from clean import OLLAMA_MODEL, clean_issue, load_issues, summarize
from models import JevModel, LayaModel, OllamaBaseline
from run_eval import QUESTION, balanced, evaluate, metrics, predict_one

ROOT = Path(__file__).parent
RUNS = ROOT / "results" / "runs"
DEFAULT_REPO = "kubernetes/kubernetes"
MODEL_COLORS = {"laya": "#4C78A8", "ollama": "#F58518", "jev": "#54A24B"}

st.set_page_config(page_title="SAD · Triagem de issues", page_icon="🐞", layout="wide")


# ================================================================ cache e utilitários

@st.cache_resource
def shared():
    """Estado do servidor, compartilhado entre reruns e com a thread de execução."""
    return {"laya": None, "lock": threading.Lock(), "job": None}


S = shared()


def get_laya():
    with S["lock"]:  # carrega uma vez só, mesmo com a thread e a página pedindo ao mesmo tempo
        if S["laya"] is None:
            m = LayaModel()
            m.ask_noul("warm-up", QUESTION)  # a 1ª chamada no MPS é lenta; fica fora das medições
            S["laya"] = m
        return S["laya"]


def make_model(name, ollama_model):
    if name == "laya":
        return get_laya()
    if name == "ollama":
        return OllamaBaseline(model=ollama_model)
    if name == "jev":
        return JevModel()
    raise ValueError(f"modelo desconhecido: {name}")  # nunca cair num modelo "padrão"


@st.cache_data(ttl=600, show_spinner="Buscando labels no GitHub...")
def repo_labels(repo):
    return fi.list_labels(fi.session(), repo)


@st.cache_data(ttl=30)
def ollama_models():
    try:
        import requests
        tags = requests.get(f"{OllamaBaseline().url}/api/tags", timeout=2).json()["models"]
        return [m["name"] for m in tags]
    except Exception:
        return []


def guess_labels(labels):
    """Sugestão inicial; o usuário confere e ajusta na tela."""
    pool = [l for l in labels if l.startswith("kind/")] or labels
    bug = [l for l in pool if re.search(r"\bbug\b|regression", l, re.I)]
    excl = [l for l in pool if re.search(r"flak|failing.?test", l, re.I)]
    nb = [l for l in pool if re.search(r"feature|enhancement|question|support|doc", l, re.I)
          and l not in bug]
    return bug, nb, excl


def dataset_file(repo):
    p = fi.dataset_path(repo)
    if not p.exists() and fi.OUT.exists():  # reaproveita o data/issues.jsonl baixado pelo CLI
        first = json.loads(fi.OUT.open().readline() or "{}")
        if f"github.com/{repo}/" in first.get("url", ""):
            p.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(fi.OUT, p)
    return p


def list_runs():
    runs = []
    for d in sorted(RUNS.glob("*"), reverse=True):
        cfg_file, csv_file = d / "config.json", d / "predictions.csv"
        if cfg_file.exists() and csv_file.exists():
            cfg = json.loads(cfg_file.read_text())
            cfg["id"] = d.name
            runs.append(cfg)
    return runs


def load_run(run_id):
    df = pd.read_csv(RUNS / run_id / "predictions.csv")
    df["prob"] = pd.to_numeric(df["prob"], errors="coerce")
    return df


def apply_threshold(df, thr):
    """Recalcula a previsão com outro threshold, sem chamar os modelos de novo."""
    df = df.copy()
    ok = df["prob"].notna()
    df.loc[ok, "pred"] = (df.loc[ok, "prob"] >= thr).map({True: "bug", False: "not_bug"})
    df.loc[~ok, "pred"] = "invalid"
    return df


def auc(df):
    pos, neg = df[df.gold == "bug"].prob.dropna(), df[df.gold != "bug"].prob.dropna()
    if pos.empty or neg.empty or df.prob.nunique() <= 2:
        return float("nan")  # modelo que só responde 0/1 não tem AUC útil
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def summary_table(df):
    rows = []
    for model, g in df.groupby("model", sort=False):
        m = metrics(g.to_dict("records"))
        rows.append({"modelo": model, "issues": m["n"], "acurácia": m["acc"], "precisão": m["prec"],
                     "recall": m["rec"], "F1": m["f1"], "AUC": auc(g),
                     "inválidas": m["invalid"], "P50 (ms)": m["p50"], "P90 (ms)": m["p90"],
                     "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"]})
    return pd.DataFrame(rows)


def color_scale(models):
    return alt.Scale(domain=list(models), range=[MODEL_COLORS.get(m, "#999") for m in models])


# ================================================================ barra lateral

st.sidebar.title("🐞 SAD · Triagem")
st.sidebar.caption("Sistema 1: modelo de decisão respondendo *é bug?* para cada issue.")

st.sidebar.header("1 · Dados")
repo = st.sidebar.text_input("Repositório (owner/nome)", DEFAULT_REPO).strip()
try:
    labels = repo_labels(repo)
except Exception as e:
    labels = []
    st.sidebar.error(f"Não consegui ler as labels de {repo}: {e}")

g_bug, g_nb, g_ex = guess_labels(labels)
if repo == DEFAULT_REPO:  # o que foi validado com você para o Kubernetes
    g_bug = [l for l in fi.DEFAULT_BUG.split(",") if l in labels]
    g_nb = [l for l in fi.DEFAULT_NOT_BUG.split(",") if l in labels]
    g_ex = [l for l in fi.DEFAULT_EXCLUDE.split(",") if l in labels]
bug_labels = st.sidebar.multiselect("Labels de **bug**", labels, g_bug, key=f"bug-{repo}")
nb_labels = st.sidebar.multiselect("Labels de **não bug**", labels, g_nb, key=f"nb-{repo}")
ex_labels = st.sidebar.multiselect("Labels ambíguas (excluir)", labels, g_ex, key=f"ex-{repo}")
per_class = st.sidebar.number_input("Issues por classe", 5, 500, 50, 5)
since = st.sidebar.text_input("Criadas a partir de (janelas, em ordem)", "2026-05-01,2026-03-01",
                              help="A segunda data só é usada se faltar issue na primeira.")

ds_path = dataset_file(repo) if repo else None
issues = load_issues(ds_path) if ds_path and ds_path.exists() else []
if issues:
    nb = sum(i["gold"] == "bug" for i in issues)
    st.sidebar.success(f"Conjunto salvo: **{nb} bug + {len(issues) - nb} não bug**  \n"
                       f"{min(i['created_at'] for i in issues)[:10]} → "
                       f"{max(i['created_at'] for i in issues)[:10]}")
else:
    st.sidebar.info("Nenhum conjunto baixado para este repositório.")

if st.sidebar.button("⬇️ Baixar issues" + (" de novo" if issues else ""), width="stretch",
                     disabled=not (bug_labels and nb_labels)):
    with st.spinner(f"Baixando issues de {repo}..."):
        try:
            issues = fi.fetch(repo, int(per_class), bug_labels, nb_labels, ex_labels,
                              [d.strip() for d in since.split(",") if d.strip()], ds_path)
            st.rerun()
        except Exception as e:
            st.sidebar.error(str(e))

st.sidebar.header("2 · Avaliação")
ollama_up = OllamaBaseline.available()
choices = ["laya", "ollama"] + (["jev"] if JevModel.available() else [])
picked = st.sidebar.multiselect("Modelos", choices, [c for c in ["laya", "ollama"] if c in choices],
                                help="O Jev aparece quando TYPESAFE_API_KEY está no .env.")
if not JevModel.available():
    st.sidebar.caption("Jev desligado: sem `TYPESAFE_API_KEY` no `.env`.")
olm = ollama_models()
ollama_model = st.sidebar.selectbox("Modelo do Ollama", olm or [OLLAMA_MODEL],
                                    index=(olm.index(OLLAMA_MODEL) if OLLAMA_MODEL in olm else 0),
                                    disabled=not ollama_up)
if not ollama_up:
    st.sidebar.warning("Ollama não está respondendo. Abra o app Ollama.")
question = st.sidebar.text_area("Pergunta (Noul)", QUESTION, height=80)
threshold = st.sidebar.slider("Threshold", 0.05, 0.95, 0.5, 0.05)
use_summary = st.sidebar.checkbox("Usar resumo quando não couber no contexto (Laya)",
                                  disabled=not ollama_up)
limit = st.sidebar.number_input("Limitar a N issues (0 = todas)", 0, 1000, 0, 2)
job = S["job"]
busy = bool(job and job["status"] in ("loading", "running"))
run_clicked = st.sidebar.button("▶️ Rodar avaliação", type="primary", width="stretch",
                                disabled=not (issues and picked) or busy)


# ================================================================ execução (thread em segundo plano)

def worker(job, cfg, todo, out):
    """Roda fora do ciclo do Streamlit: mexer na página não interrompe a avaliação."""
    try:
        models = {n: make_model(n, cfg["ollama_model"]) for n in cfg["models"]}
        job["status"] = "running"

        def on_row(r, rows):
            job["rows"] = list(rows)
            pd.DataFrame(rows).to_csv(out / "predictions.csv", index=False)
            job["done"], job["current"] = len(rows), f"{r['model']} · #{r['number']} {r['title'][:55]}"

        # um modelo de cada vez: intercalar fazia Laya e Ollama disputarem a GPU (latência inflada)
        evaluate(models, todo, cfg["threshold"], cfg["summarize"], cfg["question"],
                 on_row=on_row, should_stop=job["stop"].is_set)
        job["status"] = "stopped" if job["stop"].is_set() else "done"
    except Exception:
        job["status"], job["error"] = "error", traceback.format_exc()
    cfg.update(status=job["status"], finished_at=datetime.now().strftime("%Y%m%d-%H%M%S"))
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2))


def start_job():
    todo = balanced(issues, int(limit) or None)
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = RUNS / run_id
    out.mkdir(parents=True)
    cfg = {"repo": repo, "dataset": str(ds_path.relative_to(ROOT)), "issues": len(todo),
           "models": picked, "ollama_model": ollama_model, "question": question,
           "threshold": threshold, "summarize": use_summary, "started_at": run_id,
           "status": "running"}
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2))
    job = {"run_id": run_id, "total": len(todo) * len(picked), "done": 0, "current": "", "rows": [],
           "status": "loading", "error": None, "stop": threading.Event()}
    job["thread"] = threading.Thread(target=worker, args=(job, cfg, todo, out), daemon=True)
    S["job"] = job
    st.session_state["run_id"] = run_id
    job["thread"].start()


if run_clicked:
    start_job()


@st.fragment(run_every=2)
def job_panel():
    job = S["job"]
    if not job:
        return
    status = job["status"]
    if status in ("loading", "running"):
        st.subheader(f"▶️ Execução {job['run_id']}")
        c1, c2 = st.columns([5, 1])
        if status == "loading":
            c1.info("Carregando modelos (a Laya leva alguns segundos na primeira vez)...")
        else:
            c1.progress(job["done"] / job["total"],
                        text=f"{job['done']}/{job['total']} chamadas (um modelo de cada vez) · {job['current']}")
        if c2.button("⏹ Parar", width="stretch"):
            job["stop"].set()
        if job["rows"]:
            df = pd.DataFrame(job["rows"])
            st.dataframe(summary_table(df)[["modelo", "issues", "acurácia", "precisão", "recall",
                                            "F1", "P50 (ms)"]].round(3), hide_index=True)
            st.dataframe(df.tail(6)[["number", "model", "gold", "pred", "prob", "latency_ms",
                                     "input", "title"]], hide_index=True)
        st.caption("Pode navegar pelas abas à vontade; a execução continua em segundo plano. "
                   "As abas abaixo se atualizam sozinhas quando ela terminar.")
    else:
        if not job.get("refreshed"):
            # Terminou: recarrega a página inteira uma vez para as abas mostrarem o resultado final.
            job["refreshed"] = True
            st.rerun(scope="app")
        msg = {"done": st.success, "stopped": st.warning, "error": st.error}[status]
        text = {"done": "concluída", "stopped": "parada (resultado parcial salvo)",
                "error": "falhou"}[status]
        msg(f"Execução {job['run_id']} {text}. Resultados abaixo.")
        if job["error"]:
            st.code(job["error"])


job_panel()


# ================================================================ abas

runs = list_runs()
tab_res, tab_cmp, tab_iss = st.tabs(["📊 Resultados", "📈 Comparar execuções", "🔎 Issues"])


def run_label(r):
    live = S["job"] and S["job"]["run_id"] == r["id"] and S["job"]["status"] in ("loading", "running")
    st_ = (" · em andamento" if live else
           {"done": "", "stopped": " · parada", "error": " · com erro"}.get(r.get("status"), " · incompleta"))
    mods = ",".join(r["models"])
    return (f"{r['id']} · {r['repo']} · {mods} · {r['issues']} issues"
            + (" · resumo" if r.get("summarize") else "") + st_)


with tab_res:
    if not runs:
        st.info("Nenhuma execução ainda. Configure na barra lateral e clique em **Rodar avaliação**.")
    else:
        ids = [r["id"] for r in runs]
        default = ids.index(st.session_state["run_id"]) if st.session_state.get("run_id") in ids else 0
        sel = st.selectbox("Execução", runs, index=default, format_func=run_label)
        df_raw = load_run(sel["id"])
        st.caption(f"Pergunta: *{sel['question']}* · Ollama: `{sel.get('ollama_model', '-')}` · "
                   f"threshold original {sel['threshold']}")

        thr = st.slider("Threshold para análise (recalcula sem rodar os modelos de novo)",
                        0.05, 0.95, float(sel["threshold"]), 0.05, key=f"thr-{sel['id']}")
        df = apply_threshold(df_raw, thr)
        summ = summary_table(df)
        models_in = list(summ["modelo"])

        # --- cartões
        cols = st.columns(len(models_in))
        for col, (_, m) in zip(cols, summ.iterrows()):
            with col:
                st.markdown(f"#### {m['modelo']}")
                a, b = st.columns(2)
                a.metric("Acurácia", f"{m['acurácia']:.1%}")
                b.metric("F1 (bug)", f"{m['F1']:.3f}")
                a.metric("Precisão", f"{m['precisão']:.1%}")
                b.metric("Recall", f"{m['recall']:.1%}")
                a.metric("Latência P50", f"{m['P50 (ms)']:,.0f} ms")
                b.metric("Latência P90", f"{m['P90 (ms)']:,.0f} ms")
                extra = []
                if not math.isnan(m["AUC"]):
                    extra.append(f"AUC {m['AUC']:.3f}")
                if m["inválidas"]:
                    extra.append(f"{m['inválidas']} respostas inválidas")
                inp = df[df.model == m["modelo"]]["input"].value_counts().to_dict()
                if inp.get("truncated") or inp.get("summary"):
                    extra.append(f"{inp.get('truncated', 0)} truncadas, {inp.get('summary', 0)} via resumo")
                if extra:
                    st.caption(" · ".join(extra))

        st.dataframe(summ.drop(columns=["tp", "fp", "fn", "tn"]).round(3), hide_index=True,
                     width="stretch")

        # --- gráficos
        c1, c2 = st.columns(2)
        long = summ.melt(id_vars="modelo", value_vars=["acurácia", "precisão", "recall", "F1"],
                         var_name="métrica", value_name="valor")
        c1.markdown("**Efetividade**")
        c1.altair_chart(alt.Chart(long).mark_bar().encode(
            x=alt.X("métrica:N", sort=["acurácia", "precisão", "recall", "F1"], title=None),
            xOffset="modelo:N", y=alt.Y("valor:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("modelo:N", scale=color_scale(models_in)),
            tooltip=["modelo", "métrica", alt.Tooltip("valor:Q", format=".3f")]), width="stretch")
        c2.markdown("**Latência por chamada (ms, escala log)**")
        c2.altair_chart(alt.Chart(df).mark_boxplot(extent="min-max").encode(
            x=alt.X("model:N", title=None),
            y=alt.Y("latency_ms:Q", scale=alt.Scale(type="log"), title="ms"),
            color=alt.Color("model:N", scale=color_scale(models_in), legend=None)), width="stretch")

        st.markdown("**Matriz de confusão** (linhas = rótulo real, colunas = previsão)")
        cm_cols = st.columns(len(models_in))
        for col, model in zip(cm_cols, models_in):
            g = df[df.model == model]
            cm = (g.groupby(["gold", "pred"]).size().reset_index(name="n"))
            base = alt.Chart(cm).encode(x=alt.X("pred:N", title="previsão"),
                                        y=alt.Y("gold:N", title="real"))
            col.caption(model)
            col.altair_chart((base.mark_rect().encode(color=alt.Color("n:Q", legend=None,
                                                                       scale=alt.Scale(scheme="blues")))
                              + base.mark_text(fontSize=18).encode(text="n:Q")).properties(height=200),
                             width="stretch")

        prob_models = [m for m in models_in if df_raw[df_raw.model == m].prob.nunique() > 2]
        if prob_models:
            c3, c4 = st.columns(2)
            sweep = []
            for model in prob_models:
                g0 = df_raw[df_raw.model == model]
                for t in [x / 20 for x in range(1, 20)]:
                    m = metrics(apply_threshold(g0, t).to_dict("records"))
                    for k, v in (("acurácia", m["acc"]), ("precisão", m["prec"]),
                                 ("recall", m["rec"]), ("F1", m["f1"])):
                        sweep.append({"modelo": model, "threshold": t, "métrica": k, "valor": v})
            sw = pd.DataFrame(sweep)
            c3.markdown("**Métricas × threshold** (modelos que devolvem probabilidade)")
            c3.altair_chart((alt.Chart(sw).mark_line(point=True).encode(
                x="threshold:Q", y=alt.Y("valor:Q", scale=alt.Scale(domain=[0, 1])),
                color="métrica:N", strokeDash="modelo:N",
                tooltip=["modelo", "métrica", "threshold", alt.Tooltip("valor:Q", format=".3f")])
                + alt.Chart(pd.DataFrame({"t": [thr]})).mark_rule(color="gray").encode(x="t:Q")),
                width="stretch")
            c4.markdown("**Distribuição das probabilidades** por rótulo real")
            c4.altair_chart(alt.Chart(df_raw[df_raw.model.isin(prob_models)]).mark_bar(opacity=0.7).encode(
                x=alt.X("prob:Q", bin=alt.Bin(step=0.05), title="P(bug)"),
                y=alt.Y("count():Q", stack=None, title="issues"), color="gold:N",
                row=alt.Row("model:N", title=None)).properties(height=120), width="stretch")
            st.caption("Bom sinal: barras de *bug* concentradas à direita e de *not_bug* à esquerda. "
                       "O melhor threshold escolhido olhando este gráfico é otimista; calibre em outro conjunto.")

        # --- erros e tabela completa
        st.markdown("**Erros com maior confiança** (vale abrir: às vezes o erro é do rótulo)")
        wrong = df[(df.pred != df.gold) & df.pred.isin(["bug", "not_bug"])].copy()
        wrong["confiança no erro"] = wrong.apply(lambda r: r.prob if r.pred == "bug" else 1 - r.prob, axis=1)
        wrong = wrong.sort_values("confiança no erro", ascending=False)
        link = st.column_config.LinkColumn("issue", display_text=r"issues/(\d+)")
        st.dataframe(wrong[["url", "model", "gold", "pred", "confiança no erro", "input", "title"]],
                     column_config={"url": link}, hide_index=True, width="stretch")

        with st.expander("Todas as previsões"):
            f1, f2 = st.columns(2)
            fm = f1.multiselect("Modelo", models_in, models_in)
            fo = f2.radio("Mostrar", ["todas", "acertos", "erros"], horizontal=True)
            view = df[df.model.isin(fm)]
            if fo != "todas":
                view = view[(view.pred == view.gold) == (fo == "acertos")]
            st.dataframe(view, column_config={"url": link}, hide_index=True, width="stretch")
            st.download_button("Baixar CSV", df.to_csv(index=False), f"predictions_{sel['id']}.csv")

with tab_cmp:
    if len(runs) < 1:
        st.info("Rode pelo menos uma avaliação.")
    else:
        rows = []
        for r in runs:
            s = summary_table(load_run(r["id"]))
            for _, m in s.iterrows():
                rows.append({"execução": r["id"], "repo": r["repo"], "modelo": m["modelo"],
                             "resumo": r.get("summarize", False), "issues": m["issues"],
                             "acurácia": m["acurácia"], "F1": m["F1"], "precisão": m["precisão"],
                             "recall": m["recall"], "AUC": m["AUC"], "P50 (ms)": m["P50 (ms)"],
                             "status": r.get("status")})
        cmp = pd.DataFrame(rows)
        cmp["rótulo"] = (cmp["execução"] + " · " + cmp["issues"].astype(str) + " issues"
                         + cmp["resumo"].map({True: " · resumo", False: ""}))
        st.altair_chart(alt.Chart(cmp).mark_bar().encode(
            x=alt.X("rótulo:N", title=None, sort=None, axis=alt.Axis(labelAngle=-25, labelLimit=260)),
            xOffset="modelo:N",
            y=alt.Y("acurácia:Q", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("modelo:N", scale=color_scale(sorted(cmp.modelo.unique()))),
            tooltip=["execução", "modelo", "issues", alt.Tooltip("acurácia:Q", format=".3f"),
                     alt.Tooltip("F1:Q", format=".3f"), "P50 (ms)"]).properties(height=280),
            width="stretch")
        st.dataframe(cmp.drop(columns="rótulo").round(3), hide_index=True, width="stretch")

with tab_iss:
    if not issues:
        st.info("Baixe um conjunto de issues na barra lateral.")
    else:
        idf = pd.DataFrame(issues)
        c1, c2 = st.columns([3, 1])
        q = c1.text_input("Buscar no título", "")
        gsel = c2.selectbox("Classe", ["todas", "bug", "not_bug"])
        view = idf
        if q:
            view = view[view.title.str.contains(q, case=False, regex=False)]
        if gsel != "todas":
            view = view[view.gold == gsel]
        st.dataframe(view[["number", "gold", "title", "labels", "created_at", "url"]],
                     column_config={"url": st.column_config.LinkColumn("link", display_text="abrir")},
                     hide_index=True, width="stretch", height=260)

        num = st.selectbox("Abrir issue", view["number"].tolist(),
                           format_func=lambda n: f"#{n} · {idf[idf.number == n].title.iloc[0][:90]}")
        if num:
            it = next(i for i in issues if i["number"] == num)
            st.markdown(f"### [#{num}]({it['url']}) {it['title']}")
            st.caption(f"Rótulo real: **{it['gold']}** · labels: {', '.join(it['labels'])}")
            cleaned = clean_issue(it)
            a, b = st.columns(2)
            a.markdown(f"**Original** ({len(it['body'] or '')} caracteres)")
            a.code(it["body"] or "", language="markdown", wrap_lines=True, height=420)
            b.markdown(f"**Limpo** ({len(cleaned)} caracteres) — o que o modelo recebe")
            b.code(cleaned, language="markdown", wrap_lines=True, height=420)

            x, y = st.columns(2)
            if x.button("📝 Gerar/mostrar resumo (Ollama)", disabled=not ollama_up):
                with st.spinner("Resumindo..."):
                    st.info(summarize(it, cleaned))
            if y.button("❓ Perguntar agora aos modelos selecionados", disabled=not picked):
                res = []
                for name in picked:
                    with st.spinner(f"Perguntando a {name}..."):
                        model = make_model(name, ollama_model)
                        r = predict_one(model, it, cleaned, threshold, use_summary, question)
                    res.append({"modelo": name, "P(sim)": r["prob"], "previsão": r["pred"],
                                "acertou": r["pred"] == it["gold"], "ms": r["latency_ms"],
                                "entrada": r["input"]})
                st.dataframe(pd.DataFrame(res), hide_index=True)

            hist = []
            for r in runs:
                d = load_run(r["id"])
                for _, p in d[d.number == num].iterrows():
                    hist.append({"execução": r["id"], "modelo": p.model, "P(sim)": p.prob,
                                 "previsão": p.pred, "entrada": p.input, "ms": p.latency_ms})
            if hist:
                st.markdown("**Respostas desta issue nas execuções salvas**")
                st.dataframe(pd.DataFrame(hist), hide_index=True)
