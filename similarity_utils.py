import numpy as np
import math
import networkx as nx
import pandas as pd

from Bio import pairwise2
from sklearn.preprocessing import normalize


# While reproducing baseline methods, I implemented some useful functions, especailly the similarity calculations.
# Although they are not utilized in our method, I still make them public here, in the hope that they can be useful for you.
# "paths" is a dict recording the file paths:
'''
paths = {
    'mirna_df': 'data/our_data/nodes/mirnas.tsv',
    'disease_df': 'data/our_data/nodes/diseases.tsv',
    'mrna_df': 'data/our_data/nodes/mrnas.tsv',
    'train_val_test': 'data/our_data/train_val_test.npy',
    'mim_df': 'data/our_data/edges/mirna_mrna_encori_degradome.tsv',
    'dm_df': 'data/our_data/edges/mrna_disease_disgenet.tsv',
    'disease_disease_df': 'data/our_data/edges/disease_disease.tsv',
    'human_net_df': 'data/associations/HumanNet/HumanNet-FN.tsv'
}
'''


# ====================================================================
# Before similarity calculation, we need Functions to get ids and adjs
# ====================================================================

def get_ids(paths):
    
    mirnas_df = pd.read_table(paths['mirna_df'])
    assert len(mirnas_df['Accession']) == len(mirnas_df['Accession'].unique())
    mirna_id = pd.DataFrame(data={
        'mirna_id': mirnas_df['Accession'],
        'mapped_idx': pd.RangeIndex(len(mirnas_df['Accession']))
    })
    
    diseases_df = pd.read_table(paths['disease_df'])
    assert len(diseases_df['ID']) == len(diseases_df['ID'].unique())
    disease_id = pd.DataFrame(data={
        'disease_id': diseases_df['ID'],
        'mapped_idx': pd.RangeIndex(len(diseases_df['ID']))
    })

    mrnas_df = pd.read_table(paths['mrna_df'])
    assert len(mrnas_df['ID']) == len(mrnas_df['ID'].unique())
    mrna_id = pd.DataFrame(data={
        'mrna_id': mrnas_df['ID'],
        'mapped_idx': pd.RangeIndex(len(mrnas_df['ID']))
    })

    return mirna_id, disease_id, mrna_id


def get_mid_adj(mirna_id, disease_id, paths):
    
    adj_mid_df = pd.DataFrame(
        index=mirna_id['mirna_id'].values,
        columns=disease_id['disease_id'].values
    )

    train_val_test = np.load(paths['train_val_test'], allow_pickle=True).item()
    # Merge training and validation sets to train baseline methods
    train_val_test['train']['posi'] = train_val_test['train']['posi'] + train_val_test['val']['posi']
    train_val_test['train']['nega'] = train_val_test['train']['nega'] + train_val_test['val']['nega']
    
    train_positive_samples = train_val_test['train']['posi']
    print(len(train_positive_samples))
    for mi, d in train_positive_samples:
        adj_mid_df.loc[mi, d] = 1

    adj_mid_df = adj_mid_df.fillna(0)
    
    adj_mid = adj_mid_df.values
    adj_dmi = adj_mid.T

    return train_val_test, adj_mid, adj_dmi


def get_mim_adj(mirna_id, mrna_id, paths):

    mim_df = pd.read_table(paths['mim_df'])[['ID1', 'geneEntrezID']]

    adj_mim_df = pd.DataFrame(
        index=mirna_id['mirna_id'].values,
        columns=mrna_id['mrna_id'].values
    )

    for mi, m in mim_df[['ID1', 'geneEntrezID']].values:
        adj_mim_df.loc[mi, m] = 1
    
    adj_mim = adj_mim_df.fillna(0).values

    return adj_mim


def get_dm_adj(disease_id, mrna_id, paths):

    dm_df = pd.read_table(paths['dm_df'])[['code', 'geneId']]

    adj_dm_df = pd.DataFrame(
        index=disease_id['disease_id'].values,
        columns=mrna_id['mrna_id'].values
    )

    for d, m in dm_df.values:
        adj_dm_df.loc[d, m] = 1
    
    adj_dm = adj_dm_df.fillna(0).values

    return adj_dm



# ===================================
# Functions to Calculate Similarities 
# ===================================


# Disease Semantic Similarity
# Wang J Z, Du Z, Payattakool R, et al. A new method to measure the semantic similarity of GO terms[J]. Bioinformatics, 2007, 23(10): 1274-1281.

def get_diseases_dag(disease_id, paths):
    
    diseases_dag = nx.DiGraph()
    d_node_list = disease_id['disease_id'].values.tolist()
    disease_disease_df = pd.read_table(paths['disease_disease_df'])[['ID1', 'ID2']].drop_duplicates()
    edge_list = [(edge[0], edge[1]) for edge in disease_disease_df.values]
    
    diseases_dag.add_nodes_from(d_node_list)
    diseases_dag.add_edges_from(edge_list)

    return diseases_dag, d_node_list


# semantic contribution values (type 1)
def get_SV(diseases_dag, d, w=0.5):
    
    S = diseases_dag.subgraph(nx.descendants(diseases_dag, d) | {d})
    SV = dict()
    shortest_paths = nx.shortest_path(S, source=d)
    for x in shortest_paths:
        SV[x] = math.pow(w, (len(shortest_paths[x]) - 1))
    
    return SV


# semantic contribution values (type 2)
def get_SV2(diseases_dag, d):
    
    S = diseases_dag.subgraph(nx.descendants(diseases_dag, d) | {d})
    SV = dict()
    for x in S.nodes():
        # The number of DAGs including t / the number of diseases
        SV[x] = -1 * math.log((len(nx.ancestors(diseases_dag, x)) + 1) / len(diseases_dag.nodes()))
    
    return SV


def Wang(SV_i, SV_j):

    intersection_value = 0
    for disease in (set(SV_i.keys()) & set(SV_j.keys())):
        intersection_value = intersection_value + SV_i[disease] + SV_j[disease]
    
    return intersection_value / (sum(SV_i.values()) + sum(SV_j.values()))


def get_d_sem_sim(diseases_dag, type, diseases, w=0.5):
    
    SVs = dict()
    for d in diseases:
        if type == 1:
            SVs[d] = get_SV(diseases_dag, d, w)
        elif type == 2:
            SVs[d] = get_SV2(diseases_dag, d)

    d_len = len(diseases)
    d_sem_sim = np.eye(d_len)
    for i in range(d_len):
        for j in range(i + 1, d_len):
            d_i = diseases[i]
            d_j = diseases[j]
            d_sem_sim[i, j] = d_sem_sim[j, i] = Wang(SVs[d_i], SVs[d_j])
    
    return d_sem_sim


# Gaussian Interaction Profile Kernel Similarity
# Van Laarhoven T, Nabuurs S B, Marchiori E. Gaussian interaction profile kernels for predicting drug–target interaction[J]. Bioinformatics, 2011, 27(21): 3036-3043.

def GIPK_sim(adj, node_i, node_j, multiplier):
    
    result = np.linalg.norm(adj[node_i] - adj[node_j]) ** 2
    result = math.exp(-1 * multiplier * result)
    
    return result


def get_gipk_sim(adj, gamma=1):
    
    all_edu_dists = []
    for i in range(len(adj)):
        all_edu_dists.append(np.linalg.norm(adj[i]) ** 2)
    multiplier = gamma / (sum(all_edu_dists) / len(adj))
    
    node_len = adj.shape[0]
    gipk_sim = np.eye(node_len)
    for i in range(node_len):
        for j in range(i + 1, node_len):
            gipk_sim[i, j] = gipk_sim[j, i] = GIPK_sim(adj, i, j, multiplier)
    
    return gipk_sim


# MiRNA Sequence Similarity

def get_mi_seq_sim(paths):

    mi_df = pd.read_table(paths['mirna_df'])[['Accession', 'Sequence']]
    mi_len = len(mi_df)
    mi_seq_sim = np.eye(mi_len)
    for i in range(mi_len):
        mi_i_seq = mi_df.loc[i]['Sequence']
        mi_seq_sim[i, i] = pairwise2.align.globalxx(mi_i_seq, mi_i_seq, score_only=True)
        for j in range(i + 1, mi_len):
            mi_j_seq = mi_df.loc[j]['Sequence']
            mi_seq_sim[i, j] = mi_seq_sim[j, i] = pairwise2.align.globalxx(mi_i_seq, mi_j_seq, score_only=True)
    
    return normalize(mi_seq_sim, norm='max')


# Gene Log-Likelihood Scores

def get_gene_lls_sim(mrna_id, paths):

    human_net = pd.read_table(paths['human_net_df'], index_col=False, names=['gene1', 'gene2', 'score'])
    human_net = pd.merge(human_net, mrna_id, left_on='gene1', right_on='mrna_id', how='left')[['gene1', 'mapped_idx', 'gene2', 'score']]
    human_net.rename(columns = {'mapped_idx': 'gene_idx1'}, inplace=True)
    human_net = pd.merge(human_net, mrna_id, left_on='gene2', right_on='mrna_id', how='left')[['gene1', 'gene_idx1', 'gene2', 'mapped_idx', 'score']]
    human_net.rename(columns = {'mapped_idx': 'gene_idx2'}, inplace=True)
    human_net = human_net.dropna()
    human_net['score'] = (human_net['score'] - human_net['score'].min()) / (human_net['score'].max() - human_net['score'].min())

    gene_lls_sim = np.eye(len(mrna_id))
    
    for idx, row in human_net[['gene_idx1', 'gene_idx2', 'score']].iterrows():
        gene_idx1 = int(row['gene_idx1'])
        gene_idx2 = int(row['gene_idx2'])
        score = row['score']
        gene_lls_sim[gene_idx1, gene_idx2] = gene_lls_sim[gene_idx2, gene_idx1] = score
    
    return gene_lls_sim
    

# Functional Similarity
# Pair-wise Best, Pairs Average  
# Wang D, Wang J, Lu M, et al. Inferring the human microRNA functional similarity and functional network based on microRNA-associated diseases[J]. Bioinformatics, 2010, 26(13): 1644-1650.

def PBPA(source_i, source_j, target_sim, adj_st):
    
    target_set_i = adj_st[source_i] > 0
    target_set_j = adj_st[source_j] > 0
    target_sim_ij = target_sim[target_set_i][:, target_set_j]
    ijshape = target_sim_ij.shape
    # If a source node is not associated with any target, then similarity cannot be calculated.
    if ijshape[0] == 0 or ijshape[1] == 0:
        return np.nan
    
    return (sum(np.max(target_sim_ij, axis=0)) + sum(np.max(target_sim_ij, axis=1))) / (ijshape[0] + ijshape[1])


# miRNA (-disease) functional similarity: (d_sem_sim, adj_mid)
# miRNA (-PCG) functional similarity: (gene_lls_sim, adj_mim)
# disease (-PCG) functional similarity: (gene_lls_sim, adj_dm)
def get_fuc_sim(target_sim, adj_st):
    
    source_len = adj_st.shape[0]
    source_fuc_sim = np.eye(source_len)
    for i in range(source_len):
        for j in range(i + 1, source_len):
            source_fuc_sim[i, j] = source_fuc_sim[j, i] = PBPA(i, j, target_sim, adj_st)
    
    return source_fuc_sim
