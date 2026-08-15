# Classificador Benign/Pathogenic — Clinical Substitutions (ProteinGym) + ESM-1v

Implementação do plano: score ESM-1v (masked marginal, **um único modelo** do ensemble)
como feature de entrada para uma Regressão Logística (L2), com divisão treino/validação/teste
agrupada por proteína para evitar data leakage.

## Estrutura

```
config.py          # caminhos, hiperparâmetros, nome do modelo ESM-1v usado
data_loading.py     # leitura/consolidação dos CSVs de substituições clínicas + arquivo de referência
esm_scoring.py       # scoring masked marginal via ESM-1v (com janela deslizante para seq. >1024 aa)
splitting.py         # split treino/validação/teste agrupado por proteína + GroupKFold
evaluation.py        # métricas globais, métricas por proteína (macro) e seleção de threshold
pipeline.py           # orquestra o fluxo completo (ponto de entrada)
requirements.txt
```

## Como rodar

1. Coloque a pasta `clinical_ProteinGym_substitutions/` (um CSV por DMS_id) e o arquivo
   `clinical_substitutions.csv` na raiz do projeto (ou ajuste os caminhos em `config.py`).
2. Instale as dependências no seu ambiente CUDA já configurado:
   ```bash
   pip install -r requirements.txt
   ```
3. Rode o pipeline:
   ```bash
   python pipeline.py
   ```

Os resultados (métricas globais, métricas macro por proteína, modelo final serializado)
são salvos em `resultados/`. Os scores do ESM-1v calculados são cacheados em
`cache/esm1v_scores.csv`, então reexecuções não recalculam variantes já pontuadas.

## Decisões e pontos em aberto (ajuste conforme necessário)

- **Um único modelo ESM-1v**: conforme pedido, `config.ESM1V_MODEL_NAME` usa apenas
  `esm1v_t33_650M_UR90S_1` (não o ensemble de 5). Troque essa string para usar outro membro.
- **Métrica de seleção de C**: o plano não especifica qual métrica guia a escolha de C na
  validação cruzada agrupada (`GroupKFold`); usei **AUPRC** por ser a métrica primária do
  ProteinGym para dados clínicos e por não depender de threshold. Se preferir, troque
  `config.C_SELECTION_METRIC` e ajuste `select_best_C` em `pipeline.py`.
- **Métrica de seleção de threshold**: usei **F1** por padrão (`config.THRESHOLD_SELECTION_METRIC`),
  testando os valores em `config.THRESHOLD_GRID` sobre o conjunto de validação. MCC também
  está implementado como alternativa.
- **`class_weight="balanced"`**: é ativado automaticamente se a razão entre as classes no
  treino ultrapassar `config.IMBALANCE_RATIO_THRESHOLD` (padrão 1.5). Ajuste esse limiar ou
  force `class_weight` manualmente se quiser mais controle.
- **Vínculo com o arquivo de referência**: `DMS_id` é derivado do nome do arquivo CSV
  (`Path(csv).stem`) e casado com `DMS_filename` (também reduzido ao stem) no arquivo de
  referência. Se a convenção de nomes do seu conjunto for diferente, ajuste
  `attach_reference_metadata` em `data_loading.py`.
- **Janela deslizante (sequências > 1024 aa)**: `esm_scoring._get_window` centraliza uma
  janela de `ESM_MAX_TOKENS - 2` resíduos na posição mutada, para proteínas mais longas
  que o limite de contexto do ESM-1v — mesma estratégia usada no seu trabalho anterior com
  o benchmark DMS.
- A coluna `mutated_sequence` do CSV não é usada pelo scorer: masked marginal só precisa da
  sequência selvagem (`protein_sequence`) e da identidade da mutação (`mutant`, ex. `A329V`).

---

## Extensão: embeddings WT + PCA (201 features)

Arquivos novos, que **não alteram** `pipeline.py`, `splitting.py` nem `evaluation.py` (só
adicionam funções/módulos):

```
esm_scoring.py             # + extract_wt_embedding() (função nova, resto intocado)
splitting.py                # + group_train_val_test_split_with_arrays() (função nova, resto intocado)
cache_data_loading.py        # carrega/alinha metadata.csv + embeddings.npz + esm1vScores.csv
pca_features.py               # fit/transform de PCA, sempre fit-only-no-treino
pipeline_with_embeddings.py    # orquestra o pipeline com 201 features (esm1v_score + 200 PCA)
```

**⚠️ Isso é uma extensão explícita além do plano original**, que especifica uma única
feature de entrada (`esm1v_score`). Documentado aqui para você decidir conscientemente
se entrega essa versão, a original, ou as duas como comparação.

### Como rodar

Espera os arquivos já pré-computados em `cache/`:
```
cache/metadata.csv       # row_id, protein, mutant, position, original_aa, mutant_aa, DMS_bin_score, source_file
cache/embeddings.npz     # array 'wt_embeddings', shape (N, 1280)
cache/esm1vScores.csv    # protein, mutant, DMS_id, esm1v_score
```
```bash
py pipeline.py
```
Este script **não precisa de torch/fair-esm** para treinar — só consome os arquivos de
cache já prontos. Torch só entra na inferência (`batch_predict.py`), pra pontuar
variantes novas e extrair o embedding WT de proteínas novas.

### Decisões de design e por que

- **Alinhamento por `row_id`, nunca por posição pós-merge.** `cache_data_loading.load_and_align`
  reindexa `wt_embeddings` explicitamente usando `row_id`, então mesmo que `esm1vScores.csv`
  esteja em ordem diferente de `metadata.csv` (ou algum merge reordene linhas), o embedding
  de cada linha continua correto. Testado com dados sintéticos embaralhados de propósito.
- **PCA refeito a cada fold da validação cruzada** (não só no split final treino/val/teste).
  O pedido original menciona vazamento só no nível do split principal, mas o mesmo raciocínio
  se aplica à seleção de C: se o PCA usado durante o `GroupKFold` fosse ajustado uma vez com
  todo o treino+validação, cada fold de validação estaria influenciando (via o PCA) a feature
  usada para avaliar aquele mesmo fold. `select_best_C` em `pipeline_with_embeddings.py`
  reajusta PCA + `StandardScaler` a cada fold, no mesmo espírito do que `pipeline.py` já fazia
  com o `StandardScaler`. Isso deixa a seleção de C mais lenta (PCA é refeito ~`len(C_GRID) × N_SPLITS_CV`
  vezes) mas elimina até esse vazamento mais sutil.
- **`extract_wt_embedding` (novo, em `esm_scoring.py`) é mean-pooling da última camada,
  excluindo BOS/EOS.** Essa é uma escolha razoável e comum, mas **pode não ser idêntica** ao
  método usado para gerar o `embeddings.npz` que você já tem. Se os embeddings de treino
  vieram de outro processo (camada diferente, pooling diferente, etc.), os embeddings
  calculados na inferência não serão comparáveis aos de treino — a função já loga um aviso
  disso, mas vale você confirmar contra o script original que gerou `embeddings.npz`.
- **Sequências > 1024 aa**: para o score ESM-1v (masked marginal), a janela é centralizada na
  posição da mutação (já existia). Para o embedding WT (proteína inteira, sem uma posição de
  interesse única), a sequência é truncada a partir do N-terminal — perde informação do
  C-terminal em proteínas muito longas. Vale revisar se isso bate com o método original.
- **Inferência (`batch_predict.py`) detecta automaticamente o formato do artefato** —
  `"pca" in artifact` — e usa o fluxo de 1 ou 201 features de acordo. O embedding WT é
  cacheado por `protein_sequence` dentro do lote, então variantes da mesma proteína
  reaproveitam o mesmo embedding (testado: 6 variantes de 2 proteínas → só 2 chamadas reais
  de extração de embedding).
- **Artefato salvo em `resultados/modelo_final_embeddings.joblib`** (nome diferente do
  `modelo_final.joblib` original, para não sobrescrever o modelo de 1 feature).

### Testado (sem depender de torch/fair-esm, que não estão neste ambiente de execução)

Rodei o `pipeline_with_embeddings.py` de ponta a ponta com arquivos de cache sintéticos
(50 proteínas, 791 variantes, embeddings de 1280 dimensões com sinal real inserido) — split,
alinhamento robusto a reordenação, seleção de C com PCA refeito por fold, PCA final
fit-only-no-treino (`pca.n_samples_ == len(train)`, checado via assert), treino, threshold,
avaliação e persistência do artefato, tudo executando sem erros e produzindo métricas
coerentes com o sinal sintético inserido.

## Testes realizados

Toda a lógica de split agrupado, seleção de C via `GroupKFold`, seleção de threshold e
cálculo de métricas (globais e por proteína) foi validada com dados sintéticos antes da
entrega — incluindo checagem de que nenhuma proteína aparece em mais de um conjunto. A
etapa de scoring ESM-1v depende de `torch`/`fair-esm`, que não estão disponíveis neste
ambiente de desenvolvimento; a lógica de parsing de mutações (`A329V` → aa original,
posição, aa mutante) e de janela deslizante foi testada isoladamente.