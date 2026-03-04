import os
import torch

# Пути к данным
DATA_DIR = ''
GTF_FILE = os.path.join(DATA_DIR, 'gencode.v39.annotation.gff3')
FASTA_FILE = os.path.join(DATA_DIR, 'hg38.fa')
EXPR_FILE = os.path.join(DATA_DIR, 'GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct')

# Параметры
WINDOW = 1000
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4
TEST_CHROMS = ['chr21', 'chr22']
VAL_CHROMS = ['chr19', 'chr20']
TRAIN_CHROMS = [f'chr{i}' for i in range(1, 19)] + ['chrX']
RANDOM_SEED = 42
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Другие параметры по умолчанию
DEFAULT_N_TISSUES = None
SAVE_MODEL_PATH = 'best_model.pt'
CURRENT_MODEL_PATH = 'model.pt'