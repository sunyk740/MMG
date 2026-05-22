import numpy as np
import pandas as pd
import scanpy as sc
import torch
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import norm
from scipy.optimize import linear_sum_assignment
from torch.distributions import Normal


class HomologyMapper:
    """
    A class to map homologous cell types between two species using
    model‑based expression predictions and overlap coefficients.

    Parameters
    ----------
    merged_data : AnnData
        AnnData object containing cells from both species, with
        `.obs['species']` indicating the species of each cell.
    rmge_list : dict
        Dictionary mapping species name to a model object that has
        `.model.fc()` and `.class_names`.
    species_pair : tuple or list of str, length 2
        Names of the two species to compare, e.g. ('Human','Mouse').
    cell_type_col : str, default 'SubClass'
        Column name in `merged_data.obs` that contains the original
        cell type labels.
    overlap_threshold : float, default 0.6
        Minimum required overlap (after reordering) to consider a
        column (species B cell type) as homologous.
    adjusted_diff_threshold : float, default 0.5
        Threshold used in `compute_adjusted_overlap` to skip pairs
        where the reverse overlap difference exceeds this value.
    device : str, default 'cuda'
        Device used for torch operations.
    """

    def __init__(
        self,
        merged_data,
        rmge_list,
        species_pair,
        cell_type_col='SubClass',
        overlap_threshold=0.6,
        adjusted_diff_threshold=0.5,
        device='cuda'
    ):
        self.merged_data = merged_data
        self.rmge_list = rmge_list
        self.species_A, self.species_B = species_pair
        self.cell_type_col = cell_type_col
        self.overlap_threshold = overlap_threshold
        self.adjusted_diff_threshold = adjusted_diff_threshold
        self.device = device

        # These will be filled after running
        self.df_hm_adj = None   # A -> B adjusted
        self.df_mh_adj = None   # B -> A adjusted
        self.reordered_matrix = None
        self.merged_homologous = None

    # ---------- static utility methods (can be used outside the class) ----------
    @staticmethod
    def overlap_coefficient(dist1, dist2, x_min=-100, x_max=100, n_points=10000):
        x = np.linspace(x_min, x_max, n_points)
        p = norm.pdf(x, loc=dist1.loc.item(), scale=dist1.scale.item())
        q = norm.pdf(x, loc=dist2.loc.item(), scale=dist2.scale.item())
        return np.minimum(p, q).sum() * (x[1] - x[0])

    @staticmethod
    def reorder_columns_by_row_max(df):
        new_order = []
        seen = set()
        for i in range(len(df)):
            max_col_idx = df.iloc[i].argmax()
            max_col_name = df.columns[max_col_idx]
            if max_col_name not in seen:
                new_order.append(max_col_name)
                seen.add(max_col_name)
        for col in df.columns:
            if col not in seen:
                new_order.append(col)
        return df[new_order]

    @staticmethod
    def plot_overlap_heatmap(df, title='', annot=True, cmap='RdBu_r', center=0.5,
                             save_path=None, figsize=(14,10)):
        df_reorder = HomologyMapper.reorder_columns_by_row_max(df)
        fig, ax = plt.subplots(figsize=figsize, dpi=300)
        sns.heatmap(df_reorder, cmap=cmap, center=center,
                    annot=annot, fmt='.3f' if annot else '',
                    annot_kws={'size':7, 'weight':'bold'} if annot else None,
                    linewidths=0.5, linecolor='gray',
                    cbar_kws={'label': 'Overlap coefficient', 'shrink':0.8, 'aspect':20},
                    ax=ax)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        ax.grid(False)
        plt.tight_layout()
        if title:
            plt.title(title)
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()

    @staticmethod
    def hungarian_reorder(min_matrix):
        cost = -min_matrix.values
        row_ind, col_ind = linear_sum_assignment(cost)
        all_rows = list(range(min_matrix.shape[0]))
        all_cols = list(range(min_matrix.shape[1]))
        matched_rows = list(row_ind)
        matched_cols = list(col_ind)
        unmatched_rows = [r for r in all_rows if r not in matched_rows]
        unmatched_cols = [c for c in all_cols if c not in matched_cols]
        full_row_order = unmatched_rows + matched_rows
        full_col_order = unmatched_cols + matched_cols
        return min_matrix.iloc[full_row_order[::-1], full_col_order[::-1]]

    @staticmethod
    def plot_matched_heatmap(reordered_matrix, save_path=None):
        fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
        sns.heatmap(reordered_matrix, cmap='Purples', center=0.5, annot=False,
                    linewidths=0.5, linecolor='gray',
                    cbar_kws={'label': 'Overlap coefficient', 'shrink': 0.8, 'aspect': 20},
                    ax=ax)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        ax.grid(False)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.show()

    # ---------- internal computation methods ----------
    def _get_subclass_stats(self, data_species, model_species):
        """See original `get_subclass_stats` docstring."""
        sub_data = self.merged_data[self.merged_data.obs['species'] == data_species]
        model = self.rmge_list[model_species].model
        with torch.no_grad():
            outputs = model.fc(torch.tensor(sub_data.X).to(self.device).float())
        adata = sc.AnnData(X=outputs.cpu().numpy(), obs=sub_data.obs)
        expr_df = pd.DataFrame(adata.X, index=adata.obs.index, columns=adata.var_names)
        expr_df[self.cell_type_col] = adata.obs[self.cell_type_col]
        means = expr_df.groupby(self.cell_type_col).mean()
        stds = expr_df.groupby(self.cell_type_col).std()
        row_names = means.index.tolist()
        col_names = self.rmge_list[model_species].class_names
        means.columns = col_names
        stds.columns = col_names
        return means.values, stds.values, row_names, col_names

    def _compute_base_overlap(self, means, stds, row_names, col_names):
        overlaps = []
        for i in range(len(means)):
            max_idx = means[i].argmax()
            max_dist = Normal(loc=means[i][max_idx], scale=stds[i][max_idx])
            dists = [Normal(loc=m, scale=s) for m, s in zip(means[i], stds[i])]
            row_overlap = [self.overlap_coefficient(d, max_dist) for d in dists]
            overlaps.append(row_overlap)
        return pd.DataFrame(overlaps, index=row_names, columns=col_names)

    def _compute_adjusted_overlap(self, means, stds, row_names, col_names,
                                  base_AB, base_BA, direction='forward'):
        overlaps = []
        threshold = self.adjusted_diff_threshold
        for i in range(len(means)):
            sorted_idx = np.argsort(means[i])[::-1]
            skip_idx = []
            max_mean = max_std = max_index = None
            for idx in sorted_idx:
                if direction == 'forward':
                    cond = (base_AB.values[i, idx] - base_BA.T.values[i, idx] > threshold)
                else:
                    cond = (base_AB.values[idx, i] - base_BA.T.values[idx, i] < -threshold)
                if cond:
                    skip_idx.append(idx)
                else:
                    max_index, max_mean, max_std = idx, means[i][idx], stds[i][idx]
                    break
            if max_mean is None:
                max_index = means[i].argmax()
                max_mean, max_std = means[i][max_index], stds[i][max_index]
            max_dist = Normal(loc=max_mean, scale=max_std)
            dists = [Normal(loc=m, scale=s) for m, s in zip(means[i], stds[i])]
            row_overlap = [self.overlap_coefficient(d, max_dist) for d in dists]
            for j in skip_idx:
                row_overlap[j] = np.nan
            overlaps.append(row_overlap)
        return pd.DataFrame(overlaps, index=row_names, columns=col_names)

    # ---------- main public API ----------
    def compute_homology(self):
        """
        Run the full pipeline to identify homologous cell types and produce
        a merged AnnData object with predicted matching labels.
        """
        A = self.species_A
        B = self.species_B

        # Base overlaps
        means_AB, stds_AB, rows_A, cols_B = self._get_subclass_stats(A, B)
        base_AB = self._compute_base_overlap(means_AB, stds_AB, rows_A, cols_B)

        means_BA, stds_BA, rows_B, cols_A = self._get_subclass_stats(B, A)
        base_BA = self._compute_base_overlap(means_BA, stds_BA, rows_B, cols_A)

        # Adjusted overlaps
        self.df_hm_adj = self._compute_adjusted_overlap(
            means_AB, stds_AB, rows_A, cols_B, base_AB, base_BA, direction='forward')
        self.df_mh_adj = self._compute_adjusted_overlap(
            means_BA, stds_BA, rows_B, cols_A, base_AB, base_BA, direction='backward')

        # Minimum consensus
        min_matrix = pd.DataFrame(
            np.fmin(self.df_hm_adj.values, self.df_mh_adj.T.values),
            index=self.df_hm_adj.index, columns=self.df_hm_adj.columns
        ).fillna(0)

        # Hungarian reordering
        self.reordered_matrix = self.hungarian_reorder(min_matrix)

        # Assign predictions
        data_A = self.merged_data[self.merged_data.obs['species'] == A].copy()
        data_B = self.merged_data[self.merged_data.obs['species'] == B].copy()

        # Non-homologous types (no row with overlap > threshold)
        non_homo_cols = self.reordered_matrix.columns[
            np.sum(self.reordered_matrix > self.overlap_threshold, axis=0) == 0
        ]

        col_to_row = self.reordered_matrix.idxmax(axis=0)  # best match for each B type
        data_B.obs['predict'] = data_B.obs[self.cell_type_col].astype(str).map(col_to_row)
        data_B.obs.loc[data_B.obs[self.cell_type_col].isin(non_homo_cols), 'predict'] = 'Nonhomologous'

        data_A.obs['predict'] = data_A.obs[self.cell_type_col].astype(str)
        predicted_by_B = data_B.obs['predict'].unique()
        data_A.obs.loc[~data_A.obs[self.cell_type_col].isin(predicted_by_B), 'predict'] = 'Nonhomologous'

        # Merge homologous cells
        self.merged_homologous = sc.concat([
            data_A[data_A.obs['predict'] != 'Nonhomologous'],
            data_B[data_B.obs['predict'] != 'Nonhomologous']
        ])

        return self.merged_homologous

    def plot_consensus_heatmap(self, save_path=None):
        """Plot the consensus (Hungarian reordered) overlap heatmap."""
        if self.reordered_matrix is None:
            raise RuntimeError("Call compute_homology() first.")
        self.plot_matched_heatmap(self.reordered_matrix, save_path=save_path)


class CrossSpeciesAligner:
    """
    Harmonize cross‑species expression embeddings by aligning
    query‑species outputs to reference‑species statistics per cell type,
    then perform joint UMAP visualization.

    Parameters
    ----------
    merged_data_homologous : AnnData
        AnnData object containing cells from multiple species, with
        obs columns 'species' and 'predict' (homologous cell type label).
    rmge_list : dict
        Mapping from species name to a model object that has `.model.fc()`
        and `.class_names`.
    species_list : list of str
        Species identifiers (e.g. ['Human', 'Mouse']).
    device : str, default 'cuda'
        Device for torch operations.
    predict_col : str, default 'predict'
        Column name in obs that stores the homologous cell type label.
    nonhomologous_label : str, default 'Nonhomologous'
        Label used for cells without a homologous match.
    """

    def __init__(
        self,
        merged_data_homologous,
        rmge_list,
        species_list,
        device='cuda',
        predict_col='predict',
        nonhomologous_label='Nonhomologous'
    ):
        self.merged_data = merged_data_homologous
        self.rmge_list = rmge_list
        self.species_list = species_list
        self.device = device
        self.predict_col = predict_col
        self.nonhomologous_label = nonhomologous_label
        self.combined_data = None  # will be set after alignment

    @staticmethod
    def adjust_column_stats(reference_data, target_data):
        """
        Adjust each column of target_data to match the mean and std of reference_data.
        Both are numpy arrays with the same number of columns.
        """
        assert reference_data.shape[1] == target_data.shape[1], \
            "Both matrices must have the same number of columns."
        for col in range(reference_data.shape[1]):
            mu_ref = np.mean(reference_data[:, col])
            sigma_ref = np.std(reference_data[:, col])
            mu_tgt = np.mean(target_data[:, col])
            sigma_tgt = np.std(target_data[:, col])
            if sigma_tgt == 0:
                if len(np.unique(target_data[:, col])) == 1:
                    print(f"Warning: column {col} of target_data has only one value, setting std to 1.")
                    sigma_tgt = 1
                    sigma_ref = 1
                else:
                    print(f"Warning: column {col} of target_data has zero std, skipping adjustment.")
                    continue
            target_data[:, col] = (target_data[:, col] - mu_tgt) / sigma_tgt * sigma_ref + mu_ref
        return target_data

    def align(self):
        """
        Run cross‑species expression alignment.
        Returns the harmonized AnnData object and stores it in self.combined_data.
        """
        outputs_reference_list = {}
        outputs_query_list = {}

        for species in self.species_list:
            with torch.no_grad():
                outputs = self.rmge_list[species].model.fc(
                    torch.tensor(self.merged_data.X).to(self.device).float()
                )

            ref_mask = self.merged_data.obs['species'] == species
            query_mask = ~ref_mask

            outputs_reference = outputs[ref_mask]
            outputs_query = outputs[query_mask]

            sub_ref = self.merged_data[ref_mask]
            sub_query = self.merged_data[query_mask]

            adjusted_query = outputs_query.clone()

            for subclass in sub_query.obs[self.predict_col].unique():
                if subclass == self.nonhomologous_label:
                    continue
                print(f"Processing subclass: {subclass} for species {species}")
                st_data = adjusted_query[sub_query.obs[self.predict_col] == subclass].cpu().numpy()
                sn_data = outputs_reference[sub_ref.obs[self.predict_col] == subclass].cpu().numpy()
                adjusted = self.adjust_column_stats(sn_data, st_data)
                adjusted_query[sub_query.obs[self.predict_col] == subclass] = torch.tensor(adjusted).to(self.device)

            outputs_reference_list[species] = pd.DataFrame(
                outputs_reference.cpu(),
                index=sub_ref.obs.index
            )
            outputs_query_list[species] = pd.DataFrame(
                adjusted_query.cpu(),
                index=sub_query.obs.index
            )

        # Concatenate features with species suffix
        dfs = []
        for species in self.species_list:
            ref_idx = self.merged_data.obs_names[self.merged_data.obs['species'] == species]
            query_idx = self.merged_data.obs_names[self.merged_data.obs['species'] != species]

            ref_df = pd.DataFrame(
                outputs_reference_list[species].values,
                index=ref_idx,
                columns=[f"{c}_{species}" for c in self.rmge_list[species].class_names]
            )
            query_df = pd.DataFrame(
                outputs_query_list[species].values,
                index=query_idx,
                columns=[f"{c}_{species}" for c in self.rmge_list[species].class_names]
            )
            df_species = pd.concat([ref_df, query_df]).loc[self.merged_data.obs_names]
            dfs.append(df_species)

        final_df = pd.concat(dfs, axis=1)
        self.combined_data = sc.AnnData(X=final_df.values, obs=self.merged_data.obs)
        self.combined_data.obs['Species_SubClass'] = (
            self.combined_data.obs[self.predict_col].astype(str) + '_' +
            self.combined_data.obs['species'].astype(str)
        )
        return self.combined_data

    def run_umap(
        self,
        n_comps=5,
        n_neighbors=15,
        palette_name='tab20+tab20b+tab20c',
        use_combined_data=None
    ):
        """
        Perform PCA, build neighbour graph, run UMAP on the aligned data,
        and produce plots coloured by 'predict', 'species', and per‑species subsets.

        Parameters
        ----------
        use_combined_data : AnnData, optional
            If provided, use this AnnData instead of self.combined_data.
        """
        data = use_combined_data if use_combined_data is not None else self.combined_data
        if data is None:
            raise RuntimeError("No aligned data available. Run align() first.")

        sub_data = data[data.obs[self.predict_col] != self.nonhomologous_label].copy()
        sc.pp.pca(sub_data, n_comps=n_comps)
        sc.pp.neighbors(sub_data, n_neighbors=n_neighbors)
        sc.tl.umap(sub_data)

        n_categories = sub_data.obs[self.predict_col].nunique()
        if palette_name == 'tab20+tab20b+tab20c':
            base_palette = sns.color_palette("tab20") + sns.color_palette("tab20b") + sns.color_palette("tab20c")
        else:
            base_palette = sns.color_palette(palette_name, n_categories)
        palette = base_palette[:n_categories]

        sc.pl.umap(sub_data, color=self.predict_col, palette=palette)
        sc.pl.umap(sub_data, color='species')

        for sp in sub_data.obs['species'].unique():
            sc.pl.umap(sub_data[sub_data.obs['species'] == sp],
                       color=self.predict_col, title=sp, palette=palette)