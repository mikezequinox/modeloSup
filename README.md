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

## Testes realizados

Toda a lógica de split agrupado, seleção de C via `GroupKFold`, seleção de threshold e
cálculo de métricas (globais e por proteína) foi validada com dados sintéticos antes da
entrega — incluindo checagem de que nenhuma proteína aparece em mais de um conjunto. A
etapa de scoring ESM-1v depende de `torch`/`fair-esm`, que não estão disponíveis neste
ambiente de desenvolvimento; a lógica de parsing de mutações (`A329V` → aa original,
posição, aa mutante) e de janela deslizante foi testada isoladamente.
