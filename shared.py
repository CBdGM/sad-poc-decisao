"""Estado compartilhado entre as páginas do app (Triagem e Avaliação).

A Laya ocupa ~1,7 GB: fica num único lugar, carregada uma vez por servidor, e as duas páginas
(e a thread de execução da Avaliação) usam a mesma instância.
"""
import threading

import streamlit as st

from models import JevModel, LayaModel, OllamaBaseline
from run_eval import QUESTION


@st.cache_resource
def shared():
    """Estado do servidor, compartilhado entre reruns, páginas e a thread de execução."""
    return {"laya": None, "lock": threading.Lock(), "job": None}


S = shared()


def get_laya():
    with S["lock"]:  # carrega uma vez só, mesmo com a thread e a página pedindo ao mesmo tempo
        if S["laya"] is None:
            m = LayaModel()
            m.ask_noul("warm-up", QUESTION)  # a 1ª chamada no MPS é lenta; fica fora das medições
            S["laya"] = m
        return S["laya"]


def make_model(name, ollama_model=None):
    if name == "laya":
        return get_laya()
    if name == "ollama":
        return OllamaBaseline(model=ollama_model)
    if name == "jev":
        return JevModel()
    raise ValueError(f"modelo desconhecido: {name}")  # nunca cair num modelo "padrão"
