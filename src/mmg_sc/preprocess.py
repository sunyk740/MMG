#!/usr/bin/env python3
import scanpy as sc
import torch
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from torch.distributions import Binomial
import numpy as np
from scipy.sparse import issparse
def preprocess_data(st_data, sn_data, target_sum=1e4, n_top_genes=3000,highly_variable_genes=False):
    """
    Preprocess the spatial transcriptomics (st_data) and single-nucleus RNA-seq (sn_data) datasets.
    - Removes zero-sum columns
    - Filters common genes
    - Normalizes and log-transforms the data
    - Identifies highly variable genes for sn_data
    
    Parameters:
    - st_data: AnnData object containing spatial transcriptomics data
    - sn_data: AnnData object containing single-nucleus RNA-seq data
    - target_sum: The target sum for normalization (default is 10,000)
    - n_top_genes: The number of top variable genes to select from sn_data (default is 3000)
    
    Returns:
    - st_data: Processed AnnData object for spatial transcriptomics
    - sn_data: Processed AnnData object for single-nucleus RNA-seq
    """


    # 假设 sn_data.X 是你的数据矩阵
    if issparse(sn_data.X):
        # 如果是稀疏矩阵，转换为密集矩阵
        sn_data.X = sn_data.X.toarray()

    if issparse(st_data.X):
        # 如果是稀疏矩阵，转换为密集矩阵
        st_data.X = st_data.X.toarray()

    sn_data.layers['counts']=sn_data.X.copy()
    st_data.layers['counts']=st_data.X.copy()
    sc.pp.filter_cells(st_data, min_counts=5)
    sc.pp.filter_cells(sn_data, min_counts=5)

    valid_cells = ~np.isnan(sn_data.X).any(axis=1) & (sn_data.X >= 0).all(axis=1)
    sn_data = sn_data[valid_cells]
    cell_sums = sn_data.X.sum(axis=1)

    valid_cells = cell_sums > 0
    sn_data = sn_data[valid_cells]

    valid_cells = ~np.isnan(st_data.X).any(axis=1) & (st_data.X >= 0).all(axis=1)
    st_data = st_data[valid_cells]
    cell_sums = st_data.X.sum(axis=1)

    valid_cells = cell_sums > 0
    st_data = st_data[valid_cells]

    st_genes = st_data.var_names
    sn_genes = sn_data.var_names
    common_genes = st_genes.intersection(sn_genes)

    st_data = st_data[:, common_genes].copy()
    sn_data = sn_data[:, common_genes].copy()

    sc.pp.normalize_total(sn_data, target_sum=target_sum)
    sc.pp.log1p(sn_data)

    sc.pp.normalize_total(st_data, target_sum=target_sum)
    sc.pp.log1p(st_data)

    st_data.X = st_data.X*(sn_data.X.mean(axis=0)/st_data.X.mean(axis=0))
    sn_data.X=np.array(sn_data.X)
    st_data.X=np.array(st_data.X)

    if highly_variable_genes==True:
        sc.pp.highly_variable_genes(sn_data, n_top_genes=n_top_genes)
        # Subset the sn_data to keep only the highly variable genes
        sn_data = sn_data[:, sn_data.var['highly_variable']].copy()
        st_data = st_data[:, sn_data.var_names].copy()
        
    if (sn_data.layers['counts'].mean(axis=0)==0).sum()!=0:
        valid_gene = (sn_data.layers['counts'].mean(axis=0)>0)*(st_data.layers['counts'].mean(axis=0)>0)
        sn_data = sn_data[:,valid_gene]
        st_data = st_data[:,valid_gene]
    return st_data, sn_data
def generate_noisy_data(st_data, sn_data, device):
    """
    Generate noisy data and plot the density of log_rg values.
    
    Parameters:
    - st_data: Input data containing the 'counts' layer (AnnData object)
    - sn_data: Input data containing the 'counts' layer (AnnData object)
    - device: The device for PyTorch (either 'cpu' or 'cuda')
    
    Returns:
    - noise_sn_data: Generated noisy data tensor
    """
    

        

    p_mean = np.mean(st_data.layers['counts']) / np.mean(sn_data.layers['counts'])
    pg = np.array(st_data.layers['counts'].mean(axis=0) / sn_data.layers['counts'].mean(axis=0)).squeeze()
    rg = pg / pg.mean()
    kde = gaussian_kde(np.log(rg))
    x_range = np.linspace(min(np.log(rg)), max(np.log(rg)), 1000)

    # Calculate cell counts and percentiles
    st_cell_counts = st_data.layers['counts'].sum(axis=1)
    cell_counts = sn_data.layers['counts'].sum(axis=1).squeeze()
    try:
        st_percentiles = np.percentile(st_cell_counts, np.arange(0, 100.0, 100 / len(cell_counts)))

        # Adjust the cell counts based on percentiles
        poior_cell_counts = cell_counts.copy()
        poior_cell_counts[np.argsort(cell_counts)] = st_percentiles
    except Exception as e:
        st_percentiles = np.percentile(st_cell_counts, np.arange(0, 100.0, 100 / (len(cell_counts)-0.5)))

        # Adjust the cell counts based on percentiles
        poior_cell_counts = cell_counts.copy()
        poior_cell_counts[np.argsort(cell_counts)] = st_percentiles        
    pn = poior_cell_counts / cell_counts

    # Calculate p matrix, which is used for noise generation
    p = np.expand_dims(pn, axis=1) * np.expand_dims(rg, axis=0)
    p[p > 1] = 1

    # Generate noisy data using binomial distribution
    cell_tensor = torch.tensor(sn_data.layers['counts']).to(device)
    noise_sn_data = Binomial(total_count=cell_tensor, probs=torch.tensor(p).to(device)).sample()

    # Normalize the noisy data
    noise_sn_data = noise_sn_data / noise_sn_data.sum(axis=1, keepdims=True) * 10000
    noise_sn_data = torch.log(noise_sn_data + 1)  # Apply log transformation to smooth the data

    return noise_sn_data