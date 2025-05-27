import pandas as pd
import numpy as np
from gimmemotifs.fasta import Fasta
from gimmemotifs.motif import read_motifs
from gimmemotifs.scanner import Scanner
from gimmemotifs.config import DIRECT_NAME, INDIRECT_NAME
from genomepy import Genome
import numpy as np
import pandas as pd
from functools import reduce
import random


def get_dict_motif2TF(motifs, factor_kind=[DIRECT_NAME]):
    """Create a dictionary mapping motifs to TFs based on the factor type."""
    return {motif.id: [factor for kind in factor_kind for factor in motif.factors[kind]] for motif in motifs}


def list2str(li):
    return ', '.join(li)

def peak_M1(peak_id):
    chr_, start, end = decompose_chrstr(peak_id)
    return chr_ + '_' + str(int(start) - 1) + '_' + end

def decompose_chrstr(peak_str):
    *chr_, start, end = peak_str.split('_')
    chr_ = '_'.join(chr_)
    return (chr_, start, end)

def motif_enrichment(peak_df, motifs, ref_genome='mm10', factor_kind=[DIRECT_NAME, INDIRECT_NAME], n_cpus=1, background_length=200, fpr=0.01, min_score=10, divide=100000):
    factor_kind = [DIRECT_NAME, INDIRECT_NAME]
    dic_motif2TFs = get_dict_motif2TF(motifs, factor_kind)
    genome_data = Genome(name=ref_genome)
    peak_df = check_peak_format(peak_df, ref_genome, show_summary=False)
    target_sequences = get_sequences(peak_df, genome_data)
    rand = np.random.RandomState(42)
    s = Scanner(ncpus=n_cpus, random_state=rand)
    s.set_motifs(motifs)
    s.set_background(genome=ref_genome, size=background_length)
    s.set_threshold(fpr=fpr)

    
    scanned_df = scan_dna_for_motifs(scanner_object=s, motifs_object=motifs, sequence_object=target_sequences, divide=divide)
    df = adjust_scanned_df(scanned_df)
    peak_df['seqname'] = peak_df['chrom'].astype(str) + '_' + peak_df['start'].astype(str) + '_' + peak_df['end'].astype(str)
    peak_df1 = peak_df[['seqname', 'geneName']].rename(columns={'geneName': 'Target'})
    df = df.merge(peak_df1, on='seqname', how='inner')
    df.drop('seqname', axis=1, inplace=True)
    df['score'] = df['score'].round(5)
    df = combine_factors(df)
    df = df.sort_values(by=['Target', 'score', 'factors']).drop_duplicates(subset=['factors', 'Target'], keep='first')
    df = get_df(df, 'factors', peak_df)
    df = df[['Source', 'Target', 'score']]
    return (df, scanned_df)

def combine_factors(df):
    if 'factors_direct' not in df.columns or 'factors_indirect' not in df.columns:
        raise ValueError("The DataFrame must contain 'factors_direct' and 'factors_indirect' columns.")
    factors_df = pd.melt(df, id_vars=[col for col in df.columns if col not in ['factors_direct', 'factors_indirect']], value_vars=['factors_direct', 'factors_indirect'], var_name='factor_type', value_name='factors')
    factors_df = factors_df.drop(columns=['factor_type']).dropna(subset=['factors'])
    return factors_df

def get_df(df, col, peak_df):
    genes = list(set(peak_df.geneName))
    genes.sort()
    df = df[df[col].isin(genes)]
    df = df.groupby([col, 'Target']).sum().reset_index()
    df = df.rename(columns={col: 'Source'})
    df = df.sort_values(by=['Source', 'Target'])
    return df

def get_sequences(peak_df, genome_data):
    target_sequences = Fasta()
    for index, row in peak_df.iterrows():
        locus = (row['start'], row['end'])
        tmp = genome_data[row['chrom']][locus[0]:locus[1]]
        name = f'{tmp.name}_{tmp.start}_{tmp.end}'
        seq = tmp.seq
        target_sequences.add(name, seq)
    target_sequences = remove_zero_seq(fasta_object=target_sequences)
    return target_sequences

def check_peak_format(peaks_df, ref_genome, genomes_dir=None, show_summary=False):
    df_decomposed = peaks_df.copy()
    n_peaks_before = df_decomposed.shape[0]
    genome_data = Genome(name=ref_genome, genomes_dir=genomes_dir)
    all_chr_list = list(genome_data.keys())
    lengths = np.abs(df_decomposed['end'] - df_decomposed['start'])
    n_threshold = 5
    df_decomposed = df_decomposed[(lengths >= n_threshold) & df_decomposed.chrom.isin(all_chr_list)]
    lengths = np.abs(df_decomposed['end'] - df_decomposed['start'])
    n_invalid_length = len(lengths[lengths < n_threshold])
    n_peaks_invalid_chr = n_peaks_before - df_decomposed.chrom.isin(all_chr_list).sum()
    n_peaks_after = df_decomposed.shape[0]
    if show_summary == True:
        print('Peaks before filtering: ', n_peaks_before)
        print('Peaks with invalid chr_name: ', n_peaks_invalid_chr)
        print('Peaks with invalid length: ', n_invalid_length)
        print('Peaks after filtering: ', n_peaks_after)
    return df_decomposed

def remove_zero_seq(fasta_object):
    """
    Remove DNA sequence with zero length
    """
    fasta = Fasta()
    for i, seq in enumerate(fasta_object.seqs):
        if seq:
            name = fasta_object.ids[i]
            fasta.add(name, seq)
    return fasta

def scan_dna_for_motifs(scanner_object, motifs_object, sequence_object, divide=100000):
    """
    This is a wrapper function to scan DNA sequences searchig for Gene motifs.

    Args:

        scanner_object (gimmemotifs.scanner): Object that do motif scan.

        motifs_object (gimmemotifs.motifs): Object that stores motif data.

        sequence_object (gimmemotifs.fasta): Object that stores sequence data.

    Returns:
        pandas.dataframe: scan results is stored in data frame.

    """
    li = []
    iter = scanner_object.scan(sequence_object, scan_rc=True, nreport=250)
    for i, result in enumerate(iter):
        seqname = sequence_object.ids[i]
        for m, matches in enumerate(result):
            motif = motifs_object[m]
            for score, pos, strand in matches:
                li.append(np.array([seqname, motif.id, list2str(motif.factors[DIRECT_NAME]), list2str(motif.factors[INDIRECT_NAME]), score, pos, strand]))
    if len(li) == 0:
        df = pd.DataFrame(columns=['seqname', 'motif_id', 'factors_direct', 'factors_indirect', 'score', 'pos', 'strand'])
    else:
        remaining = 1
        LI = []
        k = 0
        li.sort(key=lambda x: (x[0], x[1], x[3], x[4], x[5]))
        while remaining == 1:
            tmp_li = li[divide * k:min(len(li), divide * (k + 1))]
            tmp_li = np.stack(tmp_li)
            df = pd.DataFrame(tmp_li, columns=['seqname', 'motif_id', 'factors_direct', 'factors_indirect', 'score', 'pos', 'strand'])
            df.score = df.score.astype(float)
            df.pos = pd.to_numeric(df['pos'])
            df.strand = pd.to_numeric(df['strand'])
            df.seqname = list(map(peak_M1, df.seqname.values))
            LI.append(df)
            if divide * (k + 1) >= len(li):
                remaining = 0
            k += 1
        df = pd.concat(LI, axis=0)
    return df

def get_dict_motif2TF(motifs, factor_kind=[DIRECT_NAME]):
    dic_motif2TFs = {}
    for i in motifs:
        fcs = []
        for j in factor_kind:
            fcs += i.factors[j]
        dic_motif2TFs[i.id] = fcs
    return dic_motif2TFs

def clean_column(df, col):
    df[col] = df[col].str.strip()
    df[col] = df[col].str.split(',')
    df = df.explode(col)
    df[col] = df[col].str.strip()
    df = df[df[col].notnull()]
    df = df[df[col] != '']
    return df

def explode_df(df):
    df = clean_column(df, 'factors_direct')
    df = clean_column(df, 'factors_indirect')
    df = df.sort_values(by=['seqname', 'score', 'factors_direct', 'factors_indirect'])
    df.reset_index(drop=True, inplace=True)
    return df

def adjust_scanned_df(scanned_df):
    df = scanned_df[['seqname', 'factors_direct', 'factors_indirect', 'score']]
    df = explode_df(df)
    return df