"""SAD · aplicação web (Streamlit).

    streamlit run app.py

Duas páginas:
  - Triagem (views/triagem.py): issues abertas de um repositório → bug?, severidade, revisão humana.
  - Avaliação (views/avaliacao.py): compara modelos num conjunto rotulado (a PoC original).
"""
import streamlit as st

# Carrega os módulos do projeto aqui, uma vez e sempre nesta ordem, antes das páginas. Sem isso, duas
# abas abrindo ao mesmo tempo (ainda mais com a máquina lenta) podiam importar em ordens diferentes,
# e uma delas via o `shared` pela metade: "cannot import name 'make_model' from 'shared'".
import models  # noqa: E402,F401
import shared  # noqa: E402,F401
import triage  # noqa: E402,F401
import fixer  # noqa: E402,F401

st.set_page_config(page_title="SAD · Triagem de issues", page_icon="🐞", layout="wide")

st.navigation([
    st.Page("views/triagem.py", title="Triagem", icon="🩺", default=True),
    st.Page("views/avaliacao.py", title="Avaliação", icon="📊"),
]).run()
