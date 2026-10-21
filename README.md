# PoC: triagem de issues com um modelo de decisão (SAD / IF1006)

Esta PoC testa o **Sistema 1** do SAD: um *modelo de decisão* (não um LLM) que recebe o texto de uma
issue e uma pergunta tipada e devolve uma probabilidade. Aqui fazemos uma pergunta só, do tipo
**Noul** (sim/não), para cada issue:

> *"Does this issue describe a defect (bug) in the software's behavior?"*

As respostas são comparadas com as labels que os mantenedores já aplicaram (bug vs. não bug), e o
resultado é medido contra um baseline "só LLM" (o mesmo LLM local respondendo direto).

| Modelo | O que é | Onde roda |
|---|---|---|
| `laya` | Laya 0.4.12, checkpoint em inglês (ModernBERT-large, **512 tokens** de contexto) | local (MPS no M1) |
| `jev` | Jev, da TypeSafe AI, via `POST /v1/systemone` (32k tokens por pergunta) | nuvem, precisa de chave |
| `ollama` | `qwen2.5:7b` respondendo `{"is_bug": true/false}`; é o baseline | local (Ollama) |

## Como rodar

```bash
cd poc-decisao
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env        # opcional: GITHUB_TOKEN, chave do Jev
```

O Ollama precisa estar rodando (abra o app **Ollama** ou rode `ollama serve`) e com o modelo baixado:

```bash
ollama pull qwen2.5:7b
```

Depois, em ordem:

```bash
.venv/bin/python fetch_issues.py --list-labels     # ver as labels do repositório
.venv/bin/python fetch_issues.py                   # baixa 50 bug + 50 não bug -> data/issues.jsonl
.venv/bin/python clean.py --show 2 --summarize     # ver a limpeza (antes/depois + resumo)
.venv/bin/python run_eval.py --models laya,ollama  # avaliação completa
```

### Interface gráfica

```bash
.venv/bin/streamlit run app.py
```

Abre em http://localhost:8501. A interface faz o mesmo que os scripts, sem terminal:

- **Barra lateral:**
  - Repositório, com as labels carregadas do GitHub e uma sugestão de quais são bug e não bug (confira).
  - Issues por classe e janelas de data; o botão **Baixar issues** guarda um conjunto por
    repositório em `data/datasets/`.
  - Modelos, modelo do Ollama, a **pergunta** (editável), threshold, resumo e limite de issues.
- **Execução:** roda em segundo plano, com barra de progresso e métricas parciais atualizadas a cada
  issue. Dá para trocar de aba enquanto ela roda, e o botão **Parar** guarda o resultado parcial.
- **📊 Resultados:**
  - Cartões com acurácia, precisão, recall, F1 e latência P50/P90 de cada modelo.
  - Gráficos de efetividade e de latência, matriz de confusão, *métricas × threshold* e a
    distribuição das probabilidades.
  - Os erros mais confiantes, com link para a issue, e o CSV completo.
  - O slider de **threshold para análise** recalcula tudo na hora, sem chamar os modelos de novo.
- **📈 Comparar execuções:** todas as execuções salvas (`results/runs/<data-hora>/`) lado a lado.
- **🔎 Issues:**
  - Busca no conjunto e mostra a issue original ao lado do texto limpo que o modelo recebe.
  - Gera o resumo, **pergunta aos modelos na hora** e mostra o que cada execução respondeu para
    aquela issue.

Observações:
- A Laya e o Ollama dividem a GPU do M1. Rodando juntos, a latência da Laya sobe (de ~0,2 s para
  ~0,5–1 s). Para medir a latência de um modelo isolado, rode-o sozinho.
- O vigia de arquivos do Streamlit está desligado (`.streamlit/config.toml`), porque ele reiniciava
  a página no meio das execuções. Depois de editar o código, reinicie o `streamlit run`.

Na primeira execução a Laya baixa os pesos do Hugging Face (~1,7 GB, fica em `~/.cache/huggingface`).
O modelo é carregado uma vez só por execução.

### Opções

`fetch_issues.py`
- `--repo owner/name` (padrão `kubernetes/kubernetes`) e `--limit N` (issues **por classe**, padrão 50).
- `--bug-labels`, `--not-bug-labels` e `--exclude-labels` recebem listas separadas por vírgula. Os
  padrões valem para o Kubernetes: bug = `kind/bug,kind/regression`; não bug =
  `kind/feature,kind/support,kind/documentation`; excluídas = `kind/flake,kind/failing-test`
  (testes instáveis ficam entre bug e não bug, por isso saem). Issues com labels das duas classes
  também são descartadas.
- `--since 2026-05-01,2026-03-01`: janelas de data em ordem de preferência. A segunda só é usada se a
  primeira não tiver issues suficientes. Desde maio só existiam 32 "não bug" válidas, então o
  conjunto atual vai até março: **50 bug** (ago–out/2026) e **49 não bug** (mar–out/2026).
- Só baixa issues fechadas, ignora pull requests e **não baixa de novo** se `data/issues.jsonl` já
  existir (use `--force` para baixar de novo).
- Usa `GITHUB_TOKEN`; se ele não existir, usa o token do `gh` CLI.

`run_eval.py`
- `--models laya,jev,ollama` (padrão: os disponíveis; o Jev entra só se houver `TYPESAFE_API_KEY`).
- `--summarize`: quando o texto limpo não cabe no contexto de um modelo (na prática só a Laya),
  usa o resumo gerado pelo Ollama. Sem essa flag, o texto é **truncado** (a Laya vê só o início), e
  o log avisa issue por issue.
- `--threshold 0.5`: probabilidade a partir da qual a previsão vira "bug".
- `--limit N`: avalia só N issues (alternando bug e não bug).

## Limpeza da entrada (`clean.py`)

**Camada 1, regex (sempre).** Ela remove o que entregaria a resposta ou só ocupa espaço:
- Comentários HTML (`<!-- ... -->`) e tags `<details>`/`<summary>`.
- **Comandos de bot** como `/kind bug`, `/sig node`, `/priority ...`, `/triage ...`, que dizem o
  rótulo literalmente.
- **Cabeçalhos dos formulários de issue**. No Kubernetes, *What happened?* e *Kubernetes version* só
  existem no template de bug, e *What would you like to be added?* só no de feature. Deixá-los
  seria entregar a resposta. Prefixos de título como `[Bug]` e `Feature:` também saem.
- **Todo cabeçalho Markdown sai inteiro** (o texto do título também), e só o conteúdo abaixo fica.
  Títulos de formulário como *Use case/motivation* ou *What happened?* diriam a classe mesmo sem o `#`.
- **Seções de ambiente/formulário** saem inteiras, com título e conteúdo. No Kubernetes: *Kubernetes
  version*, *Cloud provider*, *OS version* etc. No Airflow: *Apache Airflow version*, *Deployment*,
  *Operating System*, *Helm*, *providers*, *Code of Conduct* e *willing to submit PR*. Para outro
  repositório, confira a coluna "Limpo" na aba Issues e acrescente as seções dele em
  `TEMPLATE_DROP_SECTION` (`clean.py`). Numa auditoria, `kubectl version` aparecia em 46% dos
  bugs e em 2% dos não bugs: era uma pista estrutural. Sem elas, o Jev não mudou, e a Laya perdeu 2
  issues de acerto (ela se apoiava um pouco nisso).
- Checkboxes (`- [ ]`), `_No response_` e imagens (viram `[image]`).
- Blocos de código e logs mantêm só as primeiras 10 linhas. O conteúdo deles fica protegido das
  outras regras: um `# comentário` dentro do código não é tratado como cabeçalho.
- Espaços normalizados. A saída é `título + "\n\n" + corpo_limpo`.

**Camada 2, resumo via Ollama (`--summarize`).** Condensa a issue em até ~120 palavras, só com
fatos. O prompt proíbe classificar e usar as palavras "bug", "defect" e "feature request". O cache
fica em `data/summaries.jsonl` e muda sozinho se o prompt ou o texto mudarem.

> ⚠️ **Cuidado com o resumo.** Num teste, uma versão anterior do prompt fez o qwen2.5:7b *inventar*
> fatos sobre a issue. Ele disse que o usuário "esperava sucesso", quando o usuário esperava um erro.
> O prompt agora proíbe inferir, e a mesma issue sai correta, mas não há garantia: use
> `clean.py --summarize` para inspecionar.

## Como ler os resultados

`run_eval.py` imprime uma tabela com os modelos lado a lado e grava `results/predictions.csv`, com
uma linha por issue e modelo: `number, title, gold, model, prob, pred, latency_ms, input, url`. A
coluna `input` diz o que o modelo viu: `clean`, `summary` ou `truncated`.

Nas métricas, "positivo" é **bug**:

| Métrica | O que mede |
|---|---|
| **acurácia** | Fração de issues classificadas corretamente (bug ou não bug). Com classes balanceadas, 0,5 é o acaso. |
| **precisão (bug)** | Das issues que o modelo chamou de bug, quantas eram bug de verdade. Precisão baixa = muitos alarmes falsos. |
| **recall (bug)** | Dos bugs reais, quantos o modelo encontrou. Recall baixo = bugs passando batido. |
| **F1 (bug)** | Média harmônica de precisão e recall, um número só para a classe bug. |
| **matriz de confusão** | Linhas = rótulo real, colunas = previsão. A diagonal são os acertos; fora dela, falsos positivos (não bug → bug) e falsos negativos (bug → não bug). |
| **latência P50 / P90** | Tempo por chamada. P50 é a mediana; P90 é o tempo que 90% das chamadas não ultrapassam (mostra a cauda lenta). A primeira chamada da Laya (aquecimento do MPS) fica fora da medição. |
| **JSON inválido** (só Ollama) | Respostas que não eram `{"is_bug": true/false}`. Elas contam como erro, e a previsão vira `invalid`. |
| **5 piores erros** | Erros com maior confiança na resposta errada. Leia esses primeiro: às vezes o erro está no rótulo, não no modelo. |

**Observações para interpretar:**
- O Ollama devolve só 1,0 ou 0,0, então não há "confiança" graduada nem como mexer no threshold. A
  Laya e o Jev devolvem probabilidades, e mudar `--threshold` troca precisão por recall.
- O rótulo dos mantenedores tem ruído. A #142309, por exemplo, está como `kind/feature` mas descreve
  uma saída faltando no `kubectl describe`, que qualquer pessoa chamaria de defeito. Um "erro" do
  modelo nesses casos não é necessariamente um erro.
- Para a Laya, ver só o começo da issue (512 tokens) é uma limitação real: compare
  `report_clean.txt` com `report_summarize.txt`.
- Com ~100 issues, uma diferença de 2–3 pontos de acurácia entre modelos está dentro do ruído.

## Resultados desta execução

Veja `results/report_clean.txt` (texto limpo, Laya truncada) e `results/report_summarize.txt` (Laya
usa o resumo quando o texto não cabe). Os números estão resumidos em [RESULTADOS.md](RESULTADOS.md).

## Auditoria (`audit.py`)

```bash
.venv/bin/python audit.py                          # offline: todas as execuções salvas
.venv/bin/python audit.py --live <RUN_ID> --n 20   # + reprodução e benchmark de latência
```

**Offline** (não chama nenhum modelo), para cada execução em `results/runs/`:
- **Estrutura:** uma linha por issue e modelo, sem duplicatas.
- **Gabarito:** confere com o arquivo de issues baixado.
- **Previsões:** `pred` é exatamente `prob ≥ threshold`.
- **Métricas:** acurácia, precisão, recall e F1 recalculados com **scikit-learn**, e P50/P90 com
  **numpy**.
- **Procedência** (execuções a partir de 21/10): `served_by` registra quem respondeu segundo o
  próprio servidor (o checkpoint e o commit da Laya, a versão do Jev, o modelo do Ollama), e o Jev
  tem um `request_id` único por chamada.
- **Latência:** avisa se a execução rodou junto com outra ou se o Ollama respondeu do cache.

**Ao vivo:**
- Pergunta de novo a cada modelo e compara com o que está salvo. Laya e Ollama repetem exatamente;
  o Jev varia até ~0,03, mas mantém a previsão.
- Bloqueia a rede durante as previsões da Laya, para provar que ela roda local.
- Mede a latência de cada modelo sozinho.

**O que a auditoria corrigiu na medição de tempo:**
- **Um modelo de cada vez.** Alternar Laya e Ollama a cada issue fazia os dois disputarem a GPU do
  M1, e a Laya aparecia com P90 de ~760 ms quando o real é ~100 ms. Agora `evaluate()` roda todas as
  issues de um modelo antes de passar ao próximo, e prepara os resumos antes de cronometrar.
- **Cache do Ollama zerado antes de medir.** O Ollama guarda o prompt em cache; avaliar de novo as
  mesmas issues o deixava 3 a 13 vezes mais rápido que numa issue nova. Agora o modelo é
  descarregado e aquecido antes de cada avaliação, e a coluna `cached_tokens` mostra quanto veio do
  cache (~16% é normal: é o cabeçalho fixo do prompt).
- **O que `latency_ms` mede:** só a chamada ao modelo de decisão (tokenização + inferência; para o
  Jev, inclui a rede). Não inclui limpar o texto, gerar o resumo nem carregar o modelo. Para o Jev,
  `server_ms` separa o tempo dentro da TypeSafe (~40 ms) do tempo de rede (~180 ms).

## Como ligar o Jev

1. Pegue uma chave em https://console.typesafe.ai/keys e coloque no `.env`:
   ```
   TYPESAFE_API_KEY=sua-chave
   TYPESAFE_BASE_URL=https://api.typesafe.ai
   TYPESAFE_MODEL=jev-latest      # ou jev-preview
   ```
2. Rode `.venv/bin/python run_eval.py --models laya,jev,ollama --limit 4` e confira se não há erro de
   parsing.
3. O parser está isolado em `parse_jev_noul()` (`models.py`). Ele lê `answers.<id>.noul` e foi
   conferido numa chamada real em 20/10/2026 (`jev-1.13.0`).
4. Se a interface já estiver aberta quando você criar o `.env`, reinicie o `streamlit run` para ela
   ler a chave.
5. **Gateway (OpenRouter, Vercel, Cloudflare):** a documentação da TypeSafe não descreve nenhum. Se
   você usar um, troque `TYPESAFE_BASE_URL` pela URL do gateway. Se ele exigir outro formato de
   autenticação ou de corpo, ajuste `JevModel.ask_noul`.

O cliente já trata 429/529 (rate limit e sobrecarga) com backoff e respeita `retry-after`. Custo:
US$ 0,042 por milhão de tokens de entrada; 100 issues custam bem menos de um centavo.

## Estrutura

```
fetch_issues.py   baixa as issues rotuladas -> data/issues.jsonl
clean.py          limpeza por regex + resumo opcional via Ollama (cache em data/summaries.jsonl)
models.py         DecisionModel.ask_noul(state, question) -> float; LayaModel, JevModel, OllamaBaseline
run_eval.py       roda tudo e imprime as métricas -> results/
app.py            interface gráfica (Streamlit) -> results/runs/<data-hora>/
```
