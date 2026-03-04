import numpy as np
import pandas as pd
import torch
from pyfaidx import Fasta
import gzip

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def extract_sequence(gene_row, fasta, window):
    chrom = gene_row['chrom']
    tss = gene_row['tss']
    start = tss - window
    end = tss + window
    if start < 1 or end > len(fasta[chrom]):
        return None
    seq = fasta[chrom][start-1:end].seq.upper()
    return seq

def one_hot_encode(seq):
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    seq = seq.replace('N', 'A').replace('n', 'A')
    indices = [mapping.get(base, 0) for base in seq]
    one_hot = np.eye(4)[indices].T
    return one_hot.astype(np.float32)

def load_genes(gtf_file, chrom_list=None, gene_type='protein_coding'):
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
                if '=' in a:
                    k, v = a.strip().split('=', 1)
                    attr_dict[k] = v.strip('"')
                else:
                    pass
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
    if df_trans.empty:
        return pd.DataFrame(columns=['gene_id', 'chrom', 'strand', 'tss'])
    idx = df_trans.groupby('gene_id')['length'].idxmax()
    genes_df = df_trans.loc[idx].copy()
    genes_df['tss'] = genes_df.apply(lambda row: row['start'] if row['strand'] == '+' else row['end'], axis=1)
    genes_df = genes_df[['gene_id', 'chrom', 'strand', 'tss']].drop_duplicates()
    return genes_df

def load_expression(expr_file, n_tissues=None):
    df = pd.read_csv(expr_file, sep='\t', skiprows=2)
    df = df.set_index('Name').drop(columns=['Description'])
    if n_tissues:
        df = df.iloc[:, :n_tissues]
    df = np.log2(df + 1)
    return df