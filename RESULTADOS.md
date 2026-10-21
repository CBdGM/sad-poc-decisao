# Resultados

> **Correção de 21/10 (auditoria):** nas tabelas abaixo, as **latências da Laya estão infladas**.
> Ela alternava com o Ollama a cada issue e os dois disputavam a GPU. Medida sozinha, a Laya leva
> **P50 ~70 ms / P90 ~100 ms**. As métricas de acerto (acurácia, F1...) foram conferidas com
> scikit-learn e estão corretas. Medição limpa nas 20 issues do Airflow (execução `20261021-112612`):
>
> | | Laya | Jev | Ollama qwen2.5:7b |
> |---|---|---|---|
> | latência P50 / P90 | **68 / 98 ms** | 221 / 237 ms (~40 ms no servidor, o resto é rede) | 1.080 / 2.982 ms |

## Execução de 19/10/2026

- **Conjunto:** `kubernetes/kubernetes`, 99 issues fechadas: 50 bug (ago–out/2026) e 49 não bug
  (mar–out/2026).
- **Pergunta:** *"Does this issue describe a defect (bug) in the software's behavior?"*,
  com threshold 0,5.
- **Modelos:** Laya 0.4.12 (checkpoint em inglês, MPS) e `qwen2.5:7b` via Ollama, num MacBook Pro M1
  com 16 GB.

| | Laya (texto truncado) | Laya (+ resumo) | Ollama qwen2.5:7b |
|---|---|---|---|
| acurácia | 0,747 | 0,778 | **0,879** |
| precisão (bug) | 0,719 | 0,759 | **0,865** |
| recall (bug) | 0,820 | 0,820 | **0,900** |
| F1 (bug) | 0,766 | 0,788 | **0,882** |
| AUC (ordenação das probabilidades) | 0,846 | 0,880 | n/a (só 0/1) |
| latência P50 / P90 | **275 / 1.002 ms** | 169 / 1.055 ms* | 2.067 / 5.569 ms |
| JSON inválido | n/a | n/a | 0 de 99 |

\* Esta latência não inclui o tempo de gerar o resumo, que vem do cache e custa alguns segundos por
issue no Ollama. Na segunda rodada o Ollama também ficou mais rápido (P50 524 ms) porque reaproveita
o cache do prompt da primeira, então para latência vale a primeira rodada.

Arquivos: `results/report_clean.txt`, `results/report_summarize.txt` e
`results/predictions_{clean,summarize}.csv`.

## Com o Jev (20/10/2026)

Mesmo conjunto (99 issues), os três modelos juntos. A Laya recebe o resumo quando o texto não cabe
nela; o Jev recebe sempre o texto limpo inteiro, já que tem 32k de contexto.

| | Laya (+ resumo) | **Jev** (`jev-1.13.0`) | Ollama qwen2.5:7b |
|---|---|---|---|
| acurácia | 0,778 | 0,818 | **0,879** |
| precisão (bug) | 0,759 | 0,758 | **0,865** |
| recall (bug) | 0,820 | **0,940** | 0,900 |
| F1 (bug) | 0,788 | 0,839 | **0,882** |
| AUC | 0,880 | **0,922** | n/a |
| latência P50 / P90 | 959 / 1.329 ms* | **249 / 279 ms** | 2.613 / 6.163 ms* |

\* A Laya e o Ollama dividiram a GPU com outra execução (microsoft/vscode, pela interface), por
isso estão mais lentos que na tabela anterior. O Jev roda na nuvem e não é afetado.

- O **Jev** ficou entre os dois em acurácia, mas tem a **melhor AUC** (0,92): as probabilidades
  dele são as que melhor separam bug de não bug. Ele também tem o maior recall (deixa passar só 3
  bugs), a latência mais baixa e mais estável, e custa quase nada (~US$ 0,04 por milhão de tokens).
- O ponto fraco do Jev são os **falsos positivos** (15): ele chama de bug muitas issues `kind/feature`
  que pedem para corrigir uma limitação. Exemplos: "permitir nomes maiores que 28 caracteres" e
  "respeitar NNN no TAS". Com AUC 0,92, um threshold mais alto deve reduzir isso (veja a curva na
  interface), mas ele precisa ser calibrado em outro conjunto.
- **6 issues foram erradas pelos três modelos**: #141731, #142309, #141112, #139166, #138987 e
  #137992. Elas são as melhores candidatas a revisar o rótulo ou a pergunta.

## O que dá para concluir

1. **O baseline só-LLM acertou mais**: 88% contra 75–78% da Laya. A Laya responde ~7× mais rápido e
   roda sem gerar texto, mas, zero-shot e em inglês, ainda não alcança o qwen 7B nesta pergunta.
2. **O contexto de 512 tokens não explica tudo.** 45 das 99 issues não cabiam e foram truncadas.
   Nessas 45, o resumo subiu a acurácia da Laya de 78% para 84%. Mas nas 54 que cabiam inteiras ela
   acerta só 72%, então o limite principal é o próprio modelo, não o corte.
3. **As probabilidades da Laya ordenam bem (AUC 0,85–0,88), mas o corte em 0,5 não é o ideal.** Com
   resumo, o melhor threshold neste conjunto foi 0,35 (81% de acurácia). Esse valor foi escolhido
   olhando o próprio conjunto de teste, então é otimista; para valer, ele precisa ser calibrado em
   outro conjunto. É isso que o Ollama não oferece: ele só responde 0 ou 1, sem como ajustar.
4. **Parte dos "erros" é ruído do rótulo.** Exemplos:
   - #142309 (feature) descreve uma saída faltando no `kubectl describe`.
   - #138987 (feature) pede para tirar um valor fixo no código.
   - #141731 e #141885 (bug) são relatórios de CVE em dependências.

   Os dois modelos erram nos mesmos lugares, o que sugere ambiguidade na pergunta ou no rótulo, mais
   do que falha do modelo.

## Próximos passos sugeridos

- Ligar o Jev (32k de contexto, sem truncar) e comparar na mesma tabela.
- Testar a pergunta com `criteria` (o que conta como "sim" e como "não", por exemplo excluindo CVE de
  dependência). A Laya e o Jev aceitam esse campo.
- Calibrar o threshold num conjunto separado (outro repositório ou outro período).
