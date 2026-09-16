import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import scanpy as sc
from utils.makeGraph import ST_preprocess, intra_exp_adj
from tqdm import tqdm
import scipy.spatial.distance as sp_distance
import pandas as pd
import os
import csv


def read_h5ad_checked(path, role):
    try:
        return sc.read_h5ad(path)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to read {role} h5ad file: {os.path.abspath(str(path))}\n"
            f"{type(exc).__name__}: {exc}\n"
            "This often means the .h5ad file is corrupted, partially written, "
            "or was copied before writing finished. Regenerate or recopy it."
        ) from exc


# class ActiveDataset(Dataset):
#     def __init__(self, data_path, node_num, scale, prev_data, seed=None, k=6, neighbor_method="MNN"):
#         super(ActiveDataset, self).__init__()
#         self.data = sc.read_h5ad(data_path)
#         self.node_num = node_num
#         self.seed = seed
#         self.scale = scale
#         self.k = k
#         self.neighbor_method = neighbor_method
#         self.prev_data = prev_data
#         if self.seed:
#             np.random.seed(self.seed)
#             shuffled_indices = np.random.permutation(self.data.n_obs)
#             self.data = self.data[shuffled_indices, :]
#         self.x, self.y, self.ex_adjs = self.build_graph()
#
#     def __len__(self):
#         return len(self.x)
#
#     def __getitem__(self, idx):
#         return self.x[idx], self.y[idx], self.ex_adjs[idx]
#
#     def build_graph(self):
#         node_x_ls = []
#         node_y_ls = []
#         adj_ls = []
#         ori_data = self.data.copy()
#         pre_data = ST_preprocess(ori_data, scale=self.scale)
#         self.prev_data.var = pre_data.var
#
#         pre_data = sc.concat([pre_data, self.prev_data])
#         pre_data.obs_names = range(len(pre_data.obs_names))
#         num_graphs = int(len(pre_data) / self.node_num)
#         #         x = self.data.X
#         #         y = np.array(self.data.obs)[:,:-1]
#         for i in tqdm(range(num_graphs), desc='Generating pseudo-graphs'):
#             #         for i in range(num_graphs):
#             node = pre_data[i * 200: (i + 1) * 200]
#             node_x, node_y = node.X, np.array(node.obs)[:, :-1]
#             ex_adj = intra_exp_adj(node, dist_method="cosine", corr_dist_neighbors=self.k,
#                                    PCA_dimensionality_reduction=False,
#                                    find_neighbor_method=self.neighbor_method)
#             ex_adj = np.array(ex_adj)
#             node_x, node_y, ex_adj = torch.tensor(node_x).float(), torch.tensor(node_y).float(), torch.tensor(
#                 ex_adj).float()
#             node_x_ls.append(node_x)
#             node_y_ls.append(node_y)
#             adj_ls.append(ex_adj)
#         return node_x_ls, node_y_ls, adj_ls

class PseudoDataset(Dataset):
    def __init__(self, data_path, node_num, scale, seed=None, k=6,
                 neighbor_method="MNN", unified_genes=None):
        super(PseudoDataset, self).__init__()
        if isinstance(data_path, str):
            self.data = read_h5ad_checked(data_path, "PseudoDataset")
        else:
            self.data = read_h5ad_checked("./pseudo_graph_tmp.h5ad", "PseudoDataset")
        self.node_num = node_num
        self.seed = seed
        self.scale = scale
        self.k = k
        self.neighbor_method = neighbor_method
        self.unified_genes = list(unified_genes) if unified_genes is not None else None
        if self.seed:
            np.random.seed(self.seed)
            shuffled_indices = np.random.permutation(self.data.n_obs)
            self.data = self.data[shuffled_indices, :]
        self.x, self.y, self.ex_adjs = self.build_graph()

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.ex_adjs[idx]

    def build_graph(self):
        node_x_ls = []
        node_y_ls = []
        adj_ls = []
        ori_data = self.data.copy()
        # 只做 scale，不做基因过滤（基因已被 StDataset 对齐过）
        pre_data = ST_preprocess(ori_data, scale=self.scale, filter_genes=False)
        if self.unified_genes is not None:
            available = set(pre_data.var_names)
            common_genes = [g for g in self.unified_genes if g in available]
            missing = len(self.unified_genes) - len(common_genes)
            if missing:
                print(f"PseudoDataset: {missing} unified genes missing from pseudo data")
            pre_data = pre_data[:, common_genes]
            print(f"PseudoDataset: using unified gene set: {len(common_genes)} genes")
        num_graphs = int(len(pre_data.X) / self.node_num)
        #         x = self.data.X
        #         y = np.array(self.data.obs)[:,:-1]
        for i in tqdm(range(num_graphs), desc='Generating pseudo-graphs'):
            #         for i in range(num_graphs):
            node = pre_data[i * self.node_num: (i + 1) * self.node_num]
            node_x = node.X
            # 只取细胞类型比例列，排除 cell_num/n_counts/n_genes
            type_cols = [c for c in node.obs.columns if c not in ('cell_num', 'n_counts', 'n_genes')]
            node_y = node.obs[type_cols].values.astype(np.float32)
            ex_adj = intra_exp_adj(node, dist_method="cosine", corr_dist_neighbors=self.k,
                                   PCA_dimensionality_reduction=False,
                                   find_neighbor_method=self.neighbor_method)
            ex_adj = np.array(ex_adj)
            node_x, node_y, ex_adj = torch.tensor(node_x).float(), torch.tensor(node_y).float(), torch.tensor(
                ex_adj).float()
            node_x_ls.append(node_x)
            node_y_ls.append(node_y)
            adj_ls.append(ex_adj)
        return node_x_ls, node_y_ls, adj_ls


class StDataset(Dataset):
    def __init__(self, data_path, location_path, hvg, scale, pseudo_st_path, spatial_dist, k=6, neighbor_method="MNN", marker_path=None, save_spot=False, sp_adj_path=None, unified_genes=None):
        super(StDataset, self).__init__()
        self.save_spot = save_spot
        self.data = read_h5ad_checked(data_path, "StDataset real ST")
        if isinstance(pseudo_st_path, str):
            self.pseudo_st_data = read_h5ad_checked(pseudo_st_path, "StDataset pseudo ST")
        else:
            self.pseudo_st_data = pseudo_st_path
        self.pseudo_st_path = pseudo_st_path
        self.k = k
        self.spatial_dist = spatial_dist
        self.scale = scale
        self.hvg = hvg
        self.neighbor_method = neighbor_method
        self.marker_path = marker_path
        self.sp_adj_path = sp_adj_path
        self.unified_genes = unified_genes  # 多数据集联合训练时的统一基因集
        # location_path 可以为 None（当提供预计算的 sp_adj 时）
        if location_path and os.path.exists(location_path):
            self.loc_mat = pd.read_csv(location_path, sep='\t')
        else:
            self.loc_mat = None
        # 先 build_graph（ST_preprocess 会过滤 spot），记录过滤后的 barcode
        self.x, self.ex_adjs = self.build_graph()
        # 构建或加载 sp_adj
        self.sp_adjs = self._load_or_build_sp_adj()
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.ex_adjs[idx], self.sp_adjs[idx]

    def _ensure_min_expr_neighbors(self, ex_adj, node_x, min_degree=6, chunk_size=256):
        """Use cosine KNN fallback so every ST spot has enough expression edges."""
        A = np.asarray(ex_adj > 0, dtype=bool)
        n_spots = A.shape[0]
        if n_spots <= 1:
            return A.astype(np.float32), 0, 0.0, n_spots

        np.fill_diagonal(A, False)
        target_degree = min(max(int(min_degree), 0), n_spots - 1)
        before_degree = A.sum(axis=1)
        before_edges = int(A.sum())
        before_avg_degree = float(before_degree.mean()) if len(before_degree) else 0.0
        before_isolated = int((before_degree == 0).sum()) if len(before_degree) else 0
        if target_degree == 0 or np.all(before_degree >= target_degree):
            return A.astype(np.float32), 0, before_avg_degree, before_isolated

        X = node_x.toarray() if hasattr(node_x, "toarray") else node_x
        X = np.asarray(X, dtype=np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        row_norm = np.linalg.norm(X, axis=1, keepdims=True)
        X_norm = X / np.maximum(row_norm, 1e-8)

        need_rows = np.where(before_degree < target_degree)[0]
        for start in range(0, len(need_rows), chunk_size):
            rows = need_rows[start:start + chunk_size]
            sim = X_norm[rows] @ X_norm.T
            for local_idx, row in enumerate(rows):
                cur_degree = int(A[row].sum())
                if cur_degree >= target_degree:
                    continue
                sim[local_idx, row] = -np.inf
                order = np.argsort(sim[local_idx])[::-1]
                for col in order:
                    if cur_degree >= target_degree:
                        break
                    if col == row or A[row, col]:
                        continue
                    A[row, col] = True
                    A[col, row] = True
                    cur_degree = int(A[row].sum())

        after_edges = int(A.sum())
        return A.astype(np.float32), after_edges - before_edges, before_avg_degree, before_isolated

    def build_graph(self):
        node_x_ls = []
        node_y_ls = []
        adj_ls = []
        ori_data = self.data.copy()
        pseudo_data = self.pseudo_st_data.copy()
        # pre_data = ST_preprocess(ori_data, highly_variable_genes=self.hvg, scale=self.scale)
        if self.marker_path:
            # Define an empty list to store the cell marker values
            cell_markers = []

            # Specify the directory path
            folder_dir = self.marker_path

            # Iterate over each file in the directory
            for filename in os.listdir(folder_dir):
                if filename.endswith('.csv'):
                    # Construct the full file path
                    file_path = os.path.join(folder_dir, filename)

                    # Read the CSV file
                    with open(file_path, 'r') as file:
                        reader = csv.DictReader(file)

                        # Iterate over each row in the CSV file
                        for row in reader:
                            # Extract the cell marker value from the 'cell marker' column
                            cell_marker = row['Cell marker']

                            # Add the cell marker value to the list if it's not already present
                            if cell_marker not in cell_markers:
                                cell_markers.append(cell_marker)
            pre_data = ST_preprocess(ori_data, highly_variable_genes=False, scale=self.scale)
            st_genes = pre_data.var_names
            common_genes = set(st_genes).intersection(set(cell_markers))
            pre_data = pre_data[:, list(common_genes)]
            print(f"Select {len(common_genes)} marker genes")
        else:
            pre_data = ST_preprocess(ori_data, highly_variable_genes=self.hvg, scale=self.scale)
        st_genes = pre_data.var_names
        pseudo_spots_genes = pseudo_data.var_names

        # if not all(gene in pseudo_spots_genes for gene in st_genes):
        #     print("Not all the genes in ST recorded in pseudo spots")
        #     common_genes = set(st_genes).intersection(set(pseudo_spots_genes))
        #     # sort common_genes so that they will be always in order
        #     pseudo_data = pseudo_data[:, list(common_genes)]
        #     pre_data = pre_data[:, list(common_genes)]
        # else:
        #     pseudo_data = pseudo_data[:, pre_data.var_names]

        st_genes_set = set(st_genes)
        pseudo_spots_genes_set = set(pseudo_spots_genes)

        # 如果提供了统一基因集，强制使用它
        if self.unified_genes is not None:
            common_genes = [g for g in self.unified_genes if g in st_genes_set and g in pseudo_spots_genes_set]
            pseudo_data = pseudo_data[:, common_genes]
            pre_data = pre_data[:, common_genes]
            print(f"Using unified gene set: {len(common_genes)} genes")
        elif not st_genes_set.issubset(pseudo_spots_genes_set):
            print("Not all the genes in ST recorded in pseudo spots")
            # Keep original order of st_genes while filtering
            common_genes = [gene for gene in st_genes if gene in pseudo_spots_genes_set]

            # Subset data (works for pandas, anndata, numpy)
            pseudo_data = pseudo_data[:, common_genes]
            pre_data = pre_data[:, common_genes]
        else:
            # Ensure column alignment (if pre_data.var_names is the correct order)
            pseudo_data = pseudo_data[:, st_genes]

        # Keep the source pseudo h5ad read-only. PseudoDataset receives
        # final_genes and performs the same alignment in memory.
        # 确保 ST 数据和 pseudo 数据基因维度完全一致
        final_genes = list(pseudo_data.var_names)
        self.final_genes = final_genes
        pre_data = pre_data[:, final_genes]
        print(f"Select {len(final_genes)} HVGs (aligned with pseudo)")

        node = pre_data
        # 记录过滤后的 barcode，供 build_dist_adj 使用
        self.filtered_barcodes = list(pre_data.obs_names)
        node_x = node.X
        ex_adj = intra_exp_adj(node, dist_method="cosine", corr_dist_neighbors=self.k, PCA_dimensionality_reduction=False,
                               find_neighbor_method=self.neighbor_method)
        ex_adj = np.array(ex_adj)
        ex_adj, fallback_edges, before_avg_degree, before_isolated = self._ensure_min_expr_neighbors(
            ex_adj, node_x, min_degree=max(6, int(self.k)))
        ex_degree = (ex_adj > 0).sum(axis=1)
        ex_edges = int((ex_adj > 0).sum())
        ex_avg_degree = float(ex_degree.mean()) if len(ex_degree) else 0.0
        ex_isolated = int((ex_degree == 0).sum()) if len(ex_degree) else 0
        print(
            f"  ex_adj: {ex_adj.shape[0]} spots, method={self.neighbor_method}, "
            f"metric=cosine, k={self.k}, edges={ex_edges}, "
            f"avg_degree={ex_avg_degree:.2f}, isolated={ex_isolated}, "
            f"fallback_edges={fallback_edges}, "
            f"before_avg_degree={before_avg_degree:.2f}, "
            f"before_isolated={before_isolated}"
        )
        node_x, ex_adj = torch.tensor(node_x).float(), torch.tensor(ex_adj).float()
        node_x_ls.append(node_x)
        adj_ls.append(ex_adj)
        return node_x_ls, adj_ls

    def _load_or_build_sp_adj(self):
        """加载预计算的 sp_adj (.npy) 或从坐标计算"""
        if self.sp_adj_path and os.path.exists(self.sp_adj_path):
            sp_adj_np = np.load(self.sp_adj_path)
            n_filtered = self.x[0].shape[0] if self.x else sp_adj_np.shape[0]
            # 如果预计算的 adj 比过滤后的 spot 数大，截取对应行列
            if sp_adj_np.shape[0] > n_filtered:
                print(f"  sp_adj loaded: {self.sp_adj_path} ({sp_adj_np.shape[0]}x{sp_adj_np.shape[0]}) -> truncated to {n_filtered}x{n_filtered}")
                sp_adj_np = sp_adj_np[:n_filtered, :n_filtered]
            else:
                print(f"  sp_adj loaded: {self.sp_adj_path} ({sp_adj_np.shape[0]}x{sp_adj_np.shape[0]})")
            sp_adj = torch.tensor(sp_adj_np).float()
            return [sp_adj]
        elif self.loc_mat is not None:
            return self.build_dist_adj()
        else:
            # 既没有 npy 也没有坐标，用单位矩阵
            n = self.x[0].shape[0]
            print(f"  [WARNING] No spatial info, using identity sp_adj ({n}x{n})")
            return [torch.eye(n).float()]

    def build_dist_adj(self):
        distance_threshold = self.spatial_dist
        loc_df = self.loc_mat.copy()

        # 用过滤后的 barcode 对齐坐标，保证 sp_adj 和 ex_adj 尺寸一致
        if hasattr(self, 'filtered_barcodes') and self.filtered_barcodes:
            # 找到 barcode 列（第一个非数值列）
            non_num = [c for c in loc_df.columns if loc_df[c].dtype == object]
            if non_num:
                barcode_col = non_num[0]
                loc_df = loc_df[loc_df[barcode_col].isin(self.filtered_barcodes)]
                # 按 filtered_barcodes 的顺序排列
                loc_df = loc_df.set_index(barcode_col).loc[self.filtered_barcodes].reset_index()
            else:
                # 没有 barcode 列，按索引截取
                loc_df = loc_df.iloc[:len(self.filtered_barcodes)]

        coord_pairs = [
            ('x', 'y'),
            ('coor_X', 'coor_Y'),
            ('scaled_x', 'scaled_y'),
            ('array_row', 'array_col'),
            ('pxl_row', 'pxl_col'),
            ('imagecol', 'imagerow'),
        ]
        coord_cols = None
        for x_col, y_col in coord_pairs:
            if x_col in loc_df.columns and y_col in loc_df.columns:
                coord_cols = [x_col, y_col]
                break
        if coord_cols is None:
            num_cols = loc_df.select_dtypes(include=[np.number]).columns.tolist()
            if len(num_cols) < 2:
                raise ValueError("No usable coordinate columns found for spatial adjacency")
            coord_cols = num_cols[-2:]

        gd_loc = loc_df[coord_cols].apply(pd.to_numeric, errors='coerce').values.astype(np.float64)
        valid = np.isfinite(gd_loc).all(axis=1)
        gd_loc = gd_loc[valid]

        n_spots = gd_loc.shape[0]
        min_spatial_neighbors = 4
        spatial_k = min(min_spatial_neighbors, max(n_spots - 1, 0))
        distance_scale = float(distance_threshold) if distance_threshold else 1.5

        if n_spots <= 1 or spatial_k == 0:
            sp_adj = np.zeros((n_spots, n_spots), dtype=np.float32)
        else:
            distances = sp_distance.squareform(sp_distance.pdist(gd_loc))
            np.fill_diagonal(distances, np.inf)

            nn_dist = np.min(distances, axis=1)
            nn_dist = nn_dist[np.isfinite(nn_dist)]
            positive_dist = distances[np.isfinite(distances) & (distances > 0)]
            nn_median = float(np.median(nn_dist)) if len(nn_dist) else 0.0
            if nn_median <= 0 and len(positive_dist):
                nn_median = float(np.median(positive_dist))
            adaptive_threshold = distance_scale * nn_median if nn_median > 0 else distance_scale

            sp_adj = np.where(distances <= adaptive_threshold, 1.0, 0.0).astype(np.float32)

            knn_idx = np.argsort(distances, axis=1)[:, :spatial_k]
            row_idx = np.arange(n_spots)[:, None]
            sp_adj[row_idx, knn_idx] = 1.0

            # Keep the spatial graph undirected like the previous distance graph.
            sp_adj = np.maximum(sp_adj, sp_adj.T)
            edge_count = int(sp_adj.sum())
            avg_degree = edge_count / max(n_spots, 1)
            print(
                f"  sp_adj: {n_spots} spots, coord_cols={coord_cols}, "
                f"nn_median={nn_median:.4f}, threshold={adaptive_threshold:.4f} "
                f"(scale={distance_scale:.3f}), min_knn={spatial_k}, "
                f"edges={edge_count}, avg_degree={avg_degree:.2f}"
            )
        sp_adj = torch.tensor(sp_adj).float()
        return [sp_adj]
