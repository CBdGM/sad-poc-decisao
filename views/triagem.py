"""Página Triagem: issues abertas de um repositório → bug?, severidade, revisão humana.

Aberta pelo app.py. Cada análise vira um arquivo em results/triage/ e pode ser reaberta depois.
"""
import json
import re
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

import fixer
import triage as tr
from clean import clean_issue
from fetch_issues import fetch_numbers, fetch_open
from models import JevModel
from shared import make_model

AUTO_FIX_MAX = 5  # limita custo e tempo por análise
SEV_ICON = {"crítica": "🔴 crítica", "alta": "🟠 alta", "média": "🟡 média", "baixa": "⚪ baixa", "": "—"}

st.title("🩺 Triagem de issues")
st.caption("Informe um repositório: o modelo de decisão classifica cada issue aberta (é bug? qual a "
           "severidade? precisa de revisão humana?) e, para os bugs graves, um LLM tradicional lê o "
           "código e sugere a correção.")

# ================================================================ entrada

models = (["jev"] if JevModel.available() else []) + ["laya"]
with st.form("triagem"):
    c1, c2, c3 = st.columns([3, 1, 1])
    repo = c1.text_input("Repositório (owner/nome)", "apache/airflow").strip()
    limit = c2.number_input("Quantas issues", 5, 100, 20, 5,
                            help="As N issues abertas mais recentes.")
    days = c3.number_input("Criadas nos últimos (dias)", 1, 365, 30)
    c4, c5, c6 = st.columns([3, 1, 1])
    skip = c4.text_input("Ignorar issues com estas labels (separadas por vírgula)", "",
                         help="Ex.: labels de issues já triadas pelos mantenedores.")
    model_name = c5.selectbox("Modelo de decisão", models,
                              help="Jev (nuvem, precisa de chave) ou Laya (local, contexto de 512 tokens).")
    threshold = c6.slider("Threshold", 0.05, 0.95, 0.5, 0.05,
                          help="P(bug) a partir da qual a issue conta como bug.")
    llm_opts = ["desligada"] + fixer.available_llms()
    numbers_txt = st.text_input(
        "Ou só estas issues (números separados por vírgula)", "",
        help="Abertas ou fechadas, ex.: 74851. Ignora os campos de quantidade, dias e labels.. "
             "Em issues já corrigidas, a sugestão de fix lê o código de antes da correção.")
    c7, c8 = st.columns([2, 3])
    fix_llm = c7.selectbox("LLM da sugestão de fix", llm_opts, index=min(1, len(llm_opts) - 1),
                           help="claude = Claude via API (precisa de ANTHROPIC_API_KEY). "
                                "ollama:* = modelo local, sem custo e bem mais fraco.")
    auto_fix = c8.checkbox(f"Sugerir fix automaticamente para bugs alta/crítica (até {AUTO_FIX_MAX})",
                           value=fixer.ClaudeLLM.available(),
                           help="Ligado por padrão só com Claude: com um modelo local cada sugestão leva "
                                "minutos. Os outros bugs têm o botão Sugerir fix no detalhe.")
    go = st.form_submit_button("Analisar", type="primary")

if go:
    with st.status(f"Analisando {repo}…", expanded=True) as status:
        try:
            labels = [l.strip() for l in skip.split(",") if l.strip()]
            nums = [int(x) for x in re.findall(r"\d+", numbers_txt)]
            if nums:
                st.write(f"Buscando as issues {', '.join(f'#{n}' for n in nums)} no GitHub…")
                issues = fetch_numbers(repo, nums)
            else:
                st.write("Buscando issues abertas no GitHub…")
                issues = fetch_open(repo, int(limit), int(days), labels)
            if not issues:
                status.update(label="Nenhuma issue encontrada" + (" (números de PR não contam)." if nums
                                                                  else " aberta nessa janela."), state="error")
                st.stop()
            st.write(f"{len(issues)} issues. Carregando o modelo {model_name}…")
            model = make_model(model_name)
            bar = st.progress(0.0)
            rows = tr.triage(model, issues, threshold,
                             on_row=lambda rs: bar.progress(len(rs) / len(issues),
                                                            text=f"{len(rs)}/{len(issues)} issues"))
            params = {"limit": int(limit), "days": int(days), "skip_labels": labels, "numeros": nums,
                      "model": model_name, "threshold": threshold, "fix_llm": fix_llm}
            path = tr.save(repo, params, rows, issues)
            st.session_state["triage_path"] = str(path)
            targets = [r for r in rows if r["classe"] == "bug" and r["severidade"] in ("alta", "crítica")]
            if fix_llm != "desligada" and auto_fix and targets:
                llm = fixer.make_llm(fix_llm)
                by_num = {i["number"]: i for i in issues}
                for r in targets[:AUTO_FIX_MAX]:
                    st.write(f"🛠️ Sugerindo fix para #{r['number']} ({fix_llm})…")
                    try:
                        fix = fixer.suggest_fix(repo, by_num[r["number"]], llm)
                    except Exception as e:  # falha registrada e mostrada, nunca trocada por outro LLM
                        fix = {"erro": str(e), "meta": {"llm": fix_llm}}
                    fixer.save_fix(path, r["number"], fix)
            status.update(label=f"Pronto: {len(rows)} issues analisadas.", state="complete", expanded=False)
        except Exception as e:  # mostra o erro em vez de derrubar a página
            status.update(label="A análise falhou.", state="error")
            st.exception(e)
            st.stop()

# ================================================================ resultado

saved = tr.list_saved()
if not saved:
    st.info("Nenhuma análise ainda. Preencha os campos acima e clique em **Analisar**.")
    st.stop()


def label(p):
    d = json.loads(p.read_text())
    nums = d["params"].get("numeros")
    what = ", ".join(f"#{n}" for n in nums) if nums else f"{len(d['rows'])} issues"
    return f"{d['id']} · {d['repo']} · {d['params']['model']} · {what}"


current = st.session_state.get("triage_path")
idx = next((i for i, p in enumerate(saved) if str(p) == current), 0)
path = st.selectbox("Análise exibida", saved, index=idx, format_func=label)
data = json.loads(Path(path).read_text())

thr = st.slider("Threshold para análise (recalcula sem chamar o modelo de novo)", 0.05, 0.95,
                float(data["params"]["threshold"]), 0.05, key=f"thr-{data['id']}")
rows = tr.sort_rows([tr.decide(r, thr) for r in data["rows"]])
df = pd.DataFrame(rows)

served = ", ".join(sorted(set(df.served_by) - {""}))
trunc = int((df.input == "truncated").sum())
st.caption(f"{data['repo']} · modelo {data['params']['model']} ({served}) · "
           + (f"issues escolhidas: {', '.join(f'#{n}' for n in data['params']['numeros'])}"
              if data["params"].get("numeros") else f"issues dos últimos {data['params']['days']} dias") + (f" · {trunc} cortadas no contexto da Laya" if trunc else ""))

bugs = df[df.classe == "bug"]
m1, m2, m3, m4 = st.columns(4)
m1.metric("Issues analisadas", len(df))
m2.metric("Bugs", len(bugs))
m3.metric("Alta ou crítica", int(bugs.severidade.isin(["alta", "crítica"]).sum()))
m4.metric("Revisão humana", int(df.revisao_humana.sum()))

f1, f2 = st.columns([2, 3])
show = f1.radio("Mostrar", ["Todas", "Só bugs", "Revisão humana"], horizontal=True)
sevs = f2.multiselect("Severidade", tr.SEVERITY[::-1], tr.SEVERITY[::-1])
view = df
if show == "Só bugs":
    view = view[view.classe == "bug"]
elif show == "Revisão humana":
    view = view[view.revisao_humana]
view = view[view.severidade.isin(sevs) | (view.severidade == "")]
fixes = data.get("fixes", {})
def _mark(f):
    if "erro" in f:
        return "⚠️ falhou"
    if "grau_confianca" in f:
        return f"🛠️ {f['grau_confianca']['grau']}%"
    return "🛠️ sugerido" if f.get("codigo_encontrado") else "🔎 incerto"


fix_mark = {n: _mark(f) for n, f in fixes.items()}
view = view.assign(sev=view.severidade.map(SEV_ICON), humano=view.revisao_humana.map({True: "👤 sim", False: ""}),
                   fix=view.number.astype(str).map(fix_mark).fillna(""))

event = st.dataframe(
    # decisões primeiro, título depois: em tela estreita o que importa fica visível sem rolar
    view[["url", "sev", "humano", "fix", "classe", "p_bug", "title", "motivos", "p_repro", "created_at"]],
    column_config={
        "url": st.column_config.LinkColumn("issue", display_text=r"issues/(\d+)", width="small"),
        "title": st.column_config.TextColumn("título", width="medium"),
        "classe": "classe",
        "p_bug": st.column_config.ProgressColumn("P(bug)", min_value=0, max_value=1, format="%.2f"),
        "sev": "severidade", "humano": "revisão humana", "motivos": "motivos",
        "fix": st.column_config.TextColumn("fix", help="🛠️ N%: há sugestão de fix, com o grau de confiança "
                                           "calculado (alta ≥ 70%, média 40–69%, baixa < 40%)"),
        "p_repro": st.column_config.ProgressColumn("P(info suficiente)", min_value=0, max_value=1, format="%.2f"),
        "created_at": "criada em"},
    hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row",
    key=f"tabela-{data['id']}")
st.download_button("Baixar CSV", view.drop(columns=["sev", "humano", "fix"]).to_csv(index=False),
                   f"triagem_{data['id']}.csv")

# ================================================================ detalhe da issue

st.subheader("Detalhe")
if view.empty:
    st.caption("Nenhuma issue com esses filtros.")
    st.stop()
# Detalhe sempre aberto: o quadradinho da tabela escolhe a issue; sem seleção, vale o seletor abaixo,
# que começa na primeira issue com sugestão de fix (clicar no título não seleciona a linha).
numbers = view.number.tolist()
sel = event.selection.rows
if sel:
    current_n = numbers[sel[0]]
else:
    with_fix = [n for n in numbers if str(n) in fixes]
    pick = st.selectbox("Issue", numbers, index=numbers.index(with_fix[0]) if with_fix else 0,
                        format_func=lambda n: f"#{n} · " + ("🛠️ " if str(n) in fixes else "")
                        + view[view.number == n].title.iloc[0][:90],
                        key=f"pick-{data['id']}",
                        help="Ou marque o quadradinho à esquerda de uma linha da tabela.")
    current_n = pick
r = view[view.number == current_n].iloc[0]
issue = data["issues"][str(r.number)]
st.markdown(f"#### [#{r.number}]({r.url}) {r.title}")
st.caption(f"labels: {r.labels or '—'} · criada em {r.created_at} · entrada {r.input} · "
           f"{r.latency_ms:.0f} ms · {r.served_by}")
d1, d2, d3 = st.columns(3)
d1.metric("P(bug)", f"{r.p_bug:.2f}", r.classe, delta_color="off")
d2.metric("Severidade", r.severidade or "—", f"P = {r.p_sev:.2f}", delta_color="off")
d3.metric("P(info suficiente)", f"{r.p_repro:.2f}")
if r.revisao_humana:
    st.warning(f"👤 Revisão humana: {r.motivos}")
sev_df = pd.DataFrame({"nível": tr.SEVERITY, "probabilidade": r.sev_probs})
st.altair_chart(alt.Chart(sev_df).mark_bar().encode(
    x=alt.X("nível:N", sort=tr.SEVERITY, title="severidade", axis=alt.Axis(labelAngle=0)),
    y=alt.Y("probabilidade:Q", scale=alt.Scale(domain=[0, 1])),
    tooltip=["nível", alt.Tooltip("probabilidade:Q", format=".2f")]).properties(height=160), width="stretch")
o1, o2 = st.columns(2)
o1.markdown("**Original**")
o1.code(issue.get("body") or "", language="markdown", wrap_lines=True, height=360)
o2.markdown("**Limpo** (o que o modelo recebeu)")
o2.code(clean_issue(issue), language="markdown", wrap_lines=True, height=360)

# ================================================================ sugestão de fix

st.subheader("🛠️ Sugestão de fix")
fix = fixes.get(str(r.number))
b1, b2 = st.columns([2, 1])
choice = b1.selectbox("LLM", fixer.available_llms() or ["(nenhum LLM disponível)"],
                      key=f"llm-{data['id']}-{r.number}", label_visibility="collapsed")
if b2.button("Gerar de novo" if fix else "Sugerir fix", disabled=not fixer.available_llms(),
             key=f"fixbtn-{data['id']}-{r.number}", width="stretch"):
    with st.spinner(f"Lendo o código de {data['repo']} e pedindo a sugestão ({choice})… "
                    "a primeira vez inclui o clone do repositório."):
        try:
            new = fixer.suggest_fix(data["repo"], issue, fixer.make_llm(choice))
        except Exception as e:
            new = {"erro": str(e), "meta": {"llm": choice}}
    fixer.save_fix(path, r.number, new)
    st.rerun()

if not fix:
    st.caption("Ainda não há sugestão para esta issue. Bugs alta/crítica recebem uma automaticamente "
               "quando a análise roda com um LLM selecionado.")
elif "erro" in fix:
    st.error(f"A sugestão falhou ({fix['meta'].get('llm')}): {fix['erro']}")
elif "sugestao" not in fix:
    st.caption("Esta sugestão está no formato antigo (com patch). Clique em **Gerar de novo** para a "
               "análise no formato atual.")
else:
    meta = fix["meta"]
    conf = fix.get("grau_confianca")
    k1, k2, k3 = st.columns(3)
    if conf:
        k1.metric("Grau de confiança", f"{conf['grau']}%", delta=conf["nivel"], delta_color="off",
                  delta_arrow="off",
                  help="Calculado pelo programa a partir de checagens (detalhe abaixo); a confiança "
                       f"declarada pelo LLM ({fix['confianca']}) não entra")
    else:
        k1.metric("Confiança", fix["confianca"], help="Declarada pelo próprio LLM")
    k2.metric("Código achado?", "sim" if fix["codigo_encontrado"] else "não",
              help="O LLM diz se os arquivos lidos contêm o código do bug")
    rounds = fix.get("rodadas", [])
    k3.metric("Tempo", f"{meta['seconds']:.0f} s",
              help="Inclui a 2ª rodada, quando houve" if len(rounds) > 1 else None)
    if len(rounds) > 1:
        r2 = rounds[1]
        if fix.get("rodada_final") == 2:
            st.info("**2ª rodada.** Na 1ª, o LLM disse que o código mostrado não tinha o bug; ele pediu para "
                    f"procurar `{'`, `'.join(r2['busca'])}` e a sugestão abaixo veio desses arquivos.")
        elif r2.get("arquivos"):
            st.caption("Houve uma 2ª rodada (o LLM pediu para procurar "
                       f"`{'`, `'.join(r2['busca'])}`), mas ele também não achou o código ali; "
                       "a sugestão abaixo é a da 1ª rodada.")
        else:
            st.caption("O LLM pediu uma 2ª rodada, mas os nomes que ele pediu não levaram a arquivos novos; "
                       "a sugestão abaixo é a da 1ª rodada.")
    diag = fix.get("diagnostico_issue")
    if diag:
        box = {"diagnostico": st.success, "localizacao": st.info, "nenhuma": st.warning}[diag["nivel"]]
        note = {"diagnostico": "Sugestões costumam ser boas quando o usuário já explicou a causa.",
                "localizacao": "O grau fica limitado a 60%: sem a causa explicada, os modelos locais "
                               "acertaram o lugar mas erraram a correção.",
                "nenhuma": "O grau fica limitado a 35%: sem pistas de código, os modelos locais erraram "
                           "todas no teste. Trate a sugestão como ponto de partida."}[diag["nivel"]]
        box(f"**📋 Issue: {diag['rotulo']}.** {note}  \n"
            + (" · ".join(diag["sinais"]) if diag["sinais"] else "nenhum sinal de código encontrado na issue"))
    st.caption(f"{meta['llm']} ({', '.join(meta['served_by'])}) · custo US$ {meta['cost_usd']:.3f} · "
               f"commit {meta['commit']}"
               + (f" (antes do [PR #{meta['antes_do_fix']['pr']}]({meta['antes_do_fix']['url']}) que corrigiu a "
                  "issue: o código lido ainda tem o bug)" if meta.get("antes_do_fix") else "")
               + f" · {meta['created_at']} · "
               "nada foi executado nem enviado ao GitHub; revise antes de usar.")
    if not fix["codigo_encontrado"]:
        st.warning("O LLM diz que o código que ele leu **não contém a causa**: trate o diagnóstico como "
                   "hipótese e veja em “Como a sugestão foi montada” o que foi lido.")
    st.markdown(f"**Diagnóstico.** {fix['diagnostico']}")
    st.markdown("**Onde está a causa**")
    st.dataframe(pd.DataFrame([{"arquivo": f"https://github.com/{data['repo']}/blob/{meta['commit']}/{a['path']}",
                                "função": a.get("funcao", ""), "motivo": a["motivo"]} for a in fix["arquivos"]]),
                 column_config={"arquivo": st.column_config.LinkColumn("arquivo", display_text=r"blob/[^/]+/(.+)$")},
                 hide_index=True, width="stretch")
    st.markdown("**Sugestão de correção**")
    st.markdown(fix["sugestao"])
    if fix.get("riscos", "").strip():
        st.markdown(f"**Riscos e alternativas.** {fix['riscos']}")
    st.markdown(f"**Como verificar.** {fix['como_verificar']}")

    if conf:
        with st.expander(f"Como o grau de confiança ({conf['grau']}%) foi calculado"):
            st.dataframe(pd.DataFrame(conf["itens"]).rename(
                columns={"item": "checagem", "obtido": "pontos"})[["checagem", "pontos", "peso"]],
                hide_index=True, width="stretch")
            if conf["limitado"]:
                st.caption(f"Limitado a {conf.get('teto', 35)}%: "
                           f"{conf.get('motivo_teto') or 'o próprio LLM diz que o código lido não contém a causa'}.")
            st.caption("Alta ≥ 70%, média 40–69%, baixa < 40%. O grau mede o quanto a sugestão está "
                       "**apoiada** no código lido e nas pistas da issue, não se ela está certa; isso só "
                       "um teste confirma.")
            st.caption(f"Motivo da confiança declarada pelo LLM: {fix['motivo_confianca']}")
    else:
        st.caption(f"Motivo da confiança: {fix['motivo_confianca']}")
    with st.expander("Como a sugestão foi montada"):
        st.markdown(f"**Hipótese do passo 1:** {fix['hipotese']}")
        st.markdown("**Pistas tiradas da issue:** " + json.dumps(
            {k: v for k, v in fix["pistas"].items() if k != "frames"}, ensure_ascii=False))
        if fix.get("frames"):
            st.markdown("**Traceback** (frames que existem no repositório; o arquivo de cada um sobe nos "
                        "candidatos e o recorte começa pela função da linha citada)")
            st.dataframe(pd.DataFrame(fix["frames"])[["path", "line", "func", "raw"]].rename(
                columns={"path": "arquivo", "line": "linha", "func": "função", "raw": "como aparece na issue"}),
                hide_index=True, width="stretch")
        if fix.get("adicionado_pelo_traceback"):
            st.caption(f"O LLM não escolheu nenhum arquivo do traceback; `{fix['adicionado_pelo_traceback']}` "
                       "(o frame mais interno) foi incluído automaticamente.")
        st.markdown("**Arquivos escolhidos pelo LLM:** " + ", ".join(fix["arquivos_escolhidos"]) +
                    (f" · inexistentes (descartados): {', '.join(fix['arquivos_inexistentes'])}"
                     if fix["arquivos_inexistentes"] else ""))
        st.markdown("**Candidatos** (busca por texto, sem LLM): os 30 melhores, com o motivo")
        st.dataframe(pd.DataFrame(fix["candidatos"]), hide_index=True, width="stretch")
        # sugestões antigas não têm estes campos: mostram só o que existia
        if fix.get("simbolos"):
            st.markdown("**Mapa de símbolos** enviado no passo 1 (classes e funções de cada candidato)")
            st.code(fix["simbolos"], language="text", height=240)
        if "definicoes" in fix:
            st.markdown(f"**Definições seguidas** no passo 2: {len(fix['definicoes'])} trechos de outros "
                        "arquivos, das funções e classes usadas pelos arquivos escolhidos")
            if fix["definicoes"]:
                st.dataframe(pd.DataFrame(fix["definicoes"]).rename(
                    columns={"name": "nome", "path": "arquivo", "line": "linha", "via": "como achou"}),
                    hide_index=True, width="stretch")
        if len(rounds) > 1:
            st.markdown("**Rodadas**")
            st.dataframe(pd.DataFrame([{
                "rodada": x["rodada"], "busca pedida pelo LLM": ", ".join(x.get("busca", [])) or "—",
                "arquivos": ", ".join(x.get("arquivos", [])) or "—",
                "código achado?": {True: "sim", False: "não"}.get(x.get("codigo_encontrado"), "—"),
                "confiança": x.get("confianca", "—"), "hipótese": x.get("hipotese", "")} for x in rounds]),
                hide_index=True, width="stretch")
        st.caption(f"mapa de símbolos: {'ligado' if meta.get('mapa_simbolos') else 'desligado'} · "
                   f"seguir definições: {'ligado' if meta.get('seguir_definicoes') else 'desligado'} · "
                   f"2ª rodada: {'ligada' if meta.get('segunda_rodada') else 'desligada'}")
        st.json(meta["steps"], expanded=False)
