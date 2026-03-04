import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from pyfaidx import Fasta
import warnings
warnings.filterwarnings('ignore')

def load_expression(expr_file, n_tissues=None):

    df = pd.read_csv(expr_file, sep='\t', skiprows=2)
    df = df.set_index('Name').drop(columns=['Description'])
    if n_tissues:
        df = df.iloc[:, :n_tissues]
    df = np.log2(df + 1)
    return df

def load_genes(gtf_file, chrom_list=None, gene_type='protein_coding'):
    import gzip
    open_func = gzip.open if gtf_file.endswith('.gz') else open
    transcripts = []
    with open_func(gtf_file, 'rt') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 9:
                continue
            chrom, source, feature, start, end, score, strand, frame, attrs = parts
            if feature != 'transcript':
                continue
            attr_dict = {}
            for a in attrs.split(';'):
                if not a.strip():
                    continue
                k, v = a.strip().split('=', 1)
                attr_dict[k] = v.strip('"')
            gene_id = attr_dict.get('gene_id')
            transcript_id = attr_dict.get('transcript_id')
            gene_type_attr = attr_dict.get('gene_type')
            if gene_type_attr != gene_type:
                continue
            if chrom_list and chrom not in chrom_list:
                continue
            transcripts.append({
                'gene_id': gene_id,
                'transcript_id': transcript_id,
                'chrom': chrom,
                'strand': strand,
                'start': int(start),
                'end': int(end),
                'length': int(end) - int(start) + 1
            })

    df_trans = pd.DataFrame(transcripts)
    idx = df_trans.groupby('gene_id')['length'].idxmax()
    genes_df = df_trans.loc[idx].copy()
    genes_df['tss'] = genes_df.apply(lambda row: row['start'] if row['strand'] == '+' else row['end'], axis=1)
    genes_df = genes_df[['gene_id', 'chrom', 'strand', 'tss']].drop_duplicates()
    return genes_df

def plot_expression_distribution(expr_df, n_tissues=10, save_path='expr_distribution.png'):

    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    for i, tissue in enumerate(expr_df.columns[:n_tissues]):
        ax = axes[i]
        data = expr_df[tissue].dropna()
        ax.hist(data, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_title(f'{tissue[:20]}')
        ax.set_xlabel('log2(TPM+1)')
        ax.set_ylabel('Density')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Гистограммы сохранены в {save_path}")

def plot_tissue_correlation(expr_df, save_path='tissue_correlation.png'):

    corr = expr_df.corr(method='pearson')
 
    g = sns.clustermap(corr, cmap='vlag', center=0, linewidths=0.5,
                       figsize=(14, 14), dendrogram_ratio=0.1,
                       cbar_pos=(0.02, 0.83, 0.03, 0.15))
    g.ax_row_dendrogram.set_visible(False)  
    plt.title('Корреляция экспрессии между тканями', y=1.08)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Тепловая карта сохранена в {save_path}")

def plot_tissue_pca(expr_df, save_path='tissue_pca.png'):

    X = expr_df.T.values
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)

    pca_df = pd.DataFrame(data=components,
                          columns=['PC1', 'PC2'],
                          index=expr_df.columns)

    tissue_types = []
    for tissue in pca_df.index:
        tissue_lower = tissue.lower()
        if 'brain' in tissue_lower:
            tissue_types.append('Brain')
        elif 'muscle' in tissue_lower or 'heart' in tissue_lower:
            tissue_types.append('Muscle/Heart')
        elif 'skin' in tissue_lower:
            tissue_types.append('Skin')
        elif 'blood' in tissue_lower or 'spleen' in tissue_lower:
            tissue_types.append('Blood/Immune')
        elif 'liver' in tissue_lower or 'pancreas' in tissue_lower:
            tissue_types.append('Digestive')
        elif 'lung' in tissue_lower:
            tissue_types.append('Lung')
        else:
            tissue_types.append('Other')
    pca_df['Tissue Type'] = tissue_types

    plt.figure(figsize=(10, 8))
    sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='Tissue Type', s=100, alpha=0.8)
    for i, tissue in enumerate(pca_df.index):
        plt.annotate(tissue[:10], (pca_df.iloc[i,0], pca_df.iloc[i,1]),
                     fontsize=8, alpha=0.7)
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})')
    plt.title('PCA тканей на основе экспрессии генов')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"PCA график сохранён в {save_path}")

def plot_gc_vs_expression(gene_df, expr_df, fasta_path, window=5000, n_samples=2000, save_path='gc_vs_expr.png'):

    if not os.path.exists(fasta_path):
        print(f"Файл {fasta_path} не найден. Пропускаем GC-анализ.")
        return

    fasta = Fasta(fasta_path)
    genes_subset = gene_df.sample(n=min(n_samples, len(gene_df)), random_state=42)

    gc_values = []
    mean_expr = []

    for idx, row in genes_subset.iterrows():
        gene_id = row['gene_id']
        if gene_id not in expr_df.index:
            continue
        chrom = row['chrom']
        tss = row['tss']
        start = tss - window
        end = tss + window
        if start < 1 or end > len(fasta[chrom]):
            continue
        seq = fasta[chrom][start-1:end].seq.upper()
        if len(seq) != 2*window + 1:
            continue

        gc = (seq.count('G') + seq.count('C')) / len(seq)
        gc_values.append(gc)
        mean_expr.append(expr_df.loc[gene_id].mean()) 



    plt.figure(figsize=(8, 6))
    plt.scatter(gc_values, mean_expr, alpha=0.3, s=10, c='darkgreen')
    plt.xlabel('GC content in window')
    plt.ylabel('Mean expression (log2 TPM+1)')
    plt.title('Зависимость средней экспрессии от GC-состава')

    z = np.polyfit(gc_values, mean_expr, 1)
    p = np.poly1d(z)
    x_sorted = np.sort(gc_values)
    plt.plot(x_sorted, p(x_sorted), "r--", lw=2, label='Linear fit')

    corr, _ = pearsonr(gc_values, mean_expr)
    plt.text(0.05, 0.95, f'Pearson R = {corr:.3f}', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', alpha=0.8), fontsize=12)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"График GC vs экспрессия сохранён в {save_path}")

def distribution(expr_df):
    import seaborn as sns
    for tissue in expr_df.columns:
        sns.kdeplot(expr_df[tissue], label=tissue[:20])
    plt.legend()
    plt.xlabel('log2(TPM+1)')
    plt.show()

if __name__ == "__main__":

    DATA_DIR = ''
    GTF_FILE = os.path.join(DATA_DIR, 'gencode.v39.annotation.gff3')
    FASTA_FILE = os.path.join(DATA_DIR, 'hg38.fa')
    EXPR_FILE = os.path.join(DATA_DIR, 'GTEx_Analysis_2017-06-05_v8_RNASeQCv1.1.9_gene_median_tpm.gct')

    np.random.seed(42)
    n_genes = 500
    n_tissues = 20
    TEST_CHROMS = ['chr21', 'chr22']          
    VAL_CHROMS = ['chr19', 'chr20']           
    TRAIN_CHROMS = [f'chr{i}' for i in range(1, 19)] + ['chrX'] 
    all_chroms = TRAIN_CHROMS + VAL_CHROMS + TEST_CHROMS
    gene_df = load_genes(GTF_FILE, chrom_list=all_chroms, gene_type='protein_coding')
    print(f"Загружено {len(gene_df)} генов.")

    print("Загрузка данных экспрессии...")
    expr_df = load_expression(EXPR_FILE, n_tissues=n_tissues)
    tissue_names = expr_df.columns.tolist()
    fasta_path = 'hg38.fa' 

    distribution(expr_df)
    plot_expression_distribution(expr_df, n_tissues=10, save_path='expr_distribution.png')
    plot_tissue_correlation(expr_df, save_path='tissue_correlation.png')
    plot_tissue_pca(expr_df, save_path='tissue_pca.png')
    plot_gc_vs_expression(gene_df, expr_df, fasta_path, window=1000, n_samples=500, save_path='gc_vs_expr.png')