"""
Configurações gerais do pipeline de classificação de patogenicidade de variantes
(Clinical Substitutions - ProteinGym) via score ESM-1v + Regressão Logística.

Ajuste os caminhos abaixo para o seu ambiente antes de rodar o pipeline.
"""
from pathlib import Path

# --------------------------------------------------------------------------- #
# Caminhos
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent

# Pasta com os CSVs de substituições clínicas (um CSV por DMS_id/proteína),
# cada um com as colunas: protein, protein_sequence, mutant, mutated_sequence, DMS_bin_score
CLINICAL_SUBSTITUTIONS_DIR = BASE_DIR / "clinical_ProteinGym_substitutions"

# Arquivo de referência com metadados por DMS_id (DMS_filename, target_seq, MSA_len, ...)
REFERENCE_FILE = BASE_DIR / "clinical_substitutions.csv"

# Cache dos scores ESM-1v já calculados, para não recomputar a cada execução
CACHE_DIR = BASE_DIR / "cache"
ESM_SCORES_CACHE = CACHE_DIR / "esm1v_scores.csv"

# Saída de métricas, modelo final etc.
OUTPUT_DIR = BASE_DIR / "resultados"

# --------------------------------------------------------------------------- #
# ESM-1v
# --------------------------------------------------------------------------- #
# O plano especifica que apenas UM dos 5 modelos do ensemble ESM-1v será usado
# (não o ensemble completo). Modelos disponíveis via `esm.pretrained`:
#   esm1v_t33_650M_UR90S_1 ... esm1v_t33_650M_UR90S_5
ESM1V_MODEL_NAME = "esm1v_t33_650M_UR90S_1"
ESM_MAX_TOKENS = 1024          # limite de contexto do modelo (inclui BOS/EOS)
ESM_DEVICE = "cuda"            # "cuda" ou "cpu" — cai para cpu automaticamente se cuda não disponível

# --------------------------------------------------------------------------- #
# Divisão dos dados (agrupada por proteína, sem sobreposição de grupos)
# --------------------------------------------------------------------------- #
RANDOM_SEED = 42
TEST_SIZE = 0.15   # fração de proteínas para teste
VAL_SIZE = 0.15    # fração de proteínas para validação (sobre o total)

# --------------------------------------------------------------------------- #
# Validação cruzada / seleção de hiperparâmetros
# --------------------------------------------------------------------------- #
N_SPLITS_CV = 5
C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
THRESHOLD_GRID = [round(i / 100, 2) for i in range(5, 100, 5)]

# Métrica usada para escolher C na validação cruzada (AUPRC é a métrica primária
# do ProteinGym para dados clínicos e não depende de threshold)
C_SELECTION_METRIC = "auprc"

# Métrica usada para escolher o threshold no conjunto de validação
THRESHOLD_SELECTION_METRIC = "f1"  # alternativas: "mcc", "precision", "recall"

# --------------------------------------------------------------------------- #
# Rótulos
# --------------------------------------------------------------------------- #
LABEL_MAP = {"Benign": 0, "Pathogenic": 1}
POSITIVE_CLASS_NAME = "Pathogenic"

# Número mínimo de variantes por proteína para entrar na avaliação por proteína
MIN_VARIANTS_PER_PROTEIN = 3

# Razão máxima entre classes no treino antes de acionar class_weight="balanced"
IMBALANCE_RATIO_THRESHOLD = 1.5
