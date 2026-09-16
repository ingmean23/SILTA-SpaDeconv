import torch
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import DistanceMetric
from sklearn.neighbors import KDTree, NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity
import random
from tqdm import tqdm


def generate_a_spot_passion(sc_exp,
                            lam,
                            max_cell_types_in_spot,
                            generation_method,
                            ):
    if generation_method == 'cell':
        cell_num = np.random.poisson(lam=lam) + 1
        cell_list = list(sc_exp.obs.index.values)
        picked_cells = random.choices(cell_list, k=cell_num)
        return sc_exp[picked_cells]
    elif generation_method == 'celltype':
        cell_num = np.random.poisson(lam=lam) + 1
        cell_type_list = list(sc_exp.obs['celltype'].unique())
        cell_type_num = random.randint(1, max_cell_types_in_spot)

        while (True):
            cell_type_list_selected = random.choices(sc_exp.obs['celltype'].value_counts().keys(), k=cell_type_num)
            if len(set(cell_type_list_selected)) == cell_type_num:
                break
        sc_exp_filter = sc_exp[sc_exp.obs['celltype'].isin(cell_type_list_selected)]

        picked_cell_type = random.choices(cell_type_list_selected, k=cell_num)
        picked_cells = []
        for i in picked_cell_type:
            data = sc_exp[sc_exp.obs['celltype'] == i]
            cell_list = list(data.obs.index.values)
            picked_cells.append(random.sample(cell_list, 1)[0])
        return sc_exp_filter[picked_cells]
    else:
        print('generation_method should be "cell" or "celltype" ')


def pseudo_spot_generation(sc_exp,
                           spot_num,
                           lam,
                           max_cell_types_in_spot,
                           generation_method,
                           n_jobs=-1
                           ):
    cell_types = sc_exp.obs['celltype'].unique()

    idx_to_word_celltype = {i: word for i, word in enumerate(cell_types)}

    word_to_idx_celltype = {word: i for i, word in enumerate(cell_types)}

    print(idx_to_word_celltype)

    print(word_to_idx_celltype)

    cell_type_num = len(sc_exp.obs['celltype'].unique())

    #     cores = multiprocessing.cpu_count()
    #     if n_jobs == -1:
    #         pool = multiprocessing.Pool(processes=cores)
    #     else:
    #         pool = multiprocessing.Pool(processes=n_jobs)
    args = [(sc_exp, lam, max_cell_types_in_spot, generation_method) for i in range(spot_num)]
    #     generated_spots = pool.starmap(generate_a_spot, tqdm(args, desc='Generating pseudo-spots'))

    generated_spots = []
    for f in tqdm(args, desc='Generating pseudo-spots'):
        #         print(f)
        generated_spots.append(generate_a_spot_passion(*f))

    pseudo_spots = []
    pseudo_spots_table = np.zeros((spot_num, sc_exp.shape[1]), dtype=float)
    pseudo_fraction_table = np.zeros((spot_num, cell_type_num), dtype=float)
    for i in range(spot_num):
        one_spot = generated_spots[i]
        pseudo_spots.append(one_spot)
        pseudo_spots_table[i] = one_spot.X.sum(axis=0)
        for j in one_spot.obs.index:
            #             type_idx = one_spot.obs.loc[j, 'cell_type_idx']
            cell_type = one_spot.obs.loc[j, 'celltype']
            #             print(cell_type)
            if type(cell_type) == str:
                type_idx = word_to_idx_celltype[cell_type]
            else:
                type_idx = cell_type.map(word_to_idx_celltype)
            pseudo_fraction_table[i, type_idx] += 1
    pseudo_spots_table = pd.DataFrame(pseudo_spots_table, columns=sc_exp.var.index.values)
    pseudo_spots = sc.AnnData(X=pseudo_spots_table.iloc[:, :].values)
    pseudo_spots.obs.index = pseudo_spots_table.index[:]
    pseudo_spots.var.index = pseudo_spots_table.columns[:]
    type_list = [idx_to_word_celltype[i] for i in range(cell_type_num)]
    pseudo_fraction_table = pd.DataFrame(pseudo_fraction_table, columns=type_list)
    pseudo_fraction_table['cell_num'] = pseudo_fraction_table.sum(axis=1)
    for i in pseudo_fraction_table.columns[:-1]:
        pseudo_fraction_table[i] = pseudo_fraction_table[i] / pseudo_fraction_table['cell_num']
    pseudo_spots.obs = pseudo_spots.obs.join(pseudo_fraction_table)

    return pseudo_spots

def ST_preprocess(ST_exp,
                  normalize=False,
                  log=False,
                  highly_variable_genes=False,
                  regress_out=False,
                  scale=False,
                  scale_max_value=None,
                  scale_zero_center=True,
                  hvg_min_mean=0.0125,
                  hvg_max_mean=3,
                  hvg_min_disp=0.5,
                  highly_variable_gene_num=None,
                  filter_genes=True,
                  ):
    adata = ST_exp.copy()
    adata.var_names_make_unique()

    # 检测数据是否已经预处理过（有负值 = 已 scale，或 scanpy 标记了 log1p）
    X_check = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    already_processed = (X_check.min() < 0) or ('log1p' in adata.uns)

    if already_processed:
        print("  [ST_preprocess] Data already preprocessed, skipping normalize/log")
        if filter_genes:
            sc.pp.filter_genes(adata, min_cells=1)
        normalize = False
        log = False
    else:
        # 原始 count 数据：正常 QC
        if filter_genes:
            sc.pp.filter_cells(adata, min_counts=1)
            sc.pp.filter_genes(adata, min_cells=1)

    # 兼容旧版scanpy: highly_variable_genes 会访问 uns['log1p']['base']
    if 'log1p' not in adata.uns:
        adata.uns['log1p'] = {'base': None}
    elif 'base' not in adata.uns['log1p']:
        adata.uns['log1p']['base'] = None

    if normalize == True:
        sc.pp.normalize_total(adata, target_sum=1e4)

    if log == True:
        sc.pp.log1p(adata)

    # 清理 inf/nan（稀疏矩阵先转dense）
    if hasattr(adata.X, "toarray"):
        adata.X = adata.X.toarray()
    adata.X = np.nan_to_num(adata.X, nan=0.0, posinf=0.0, neginf=0.0)

    adata.layers['scale.data'] = adata.X.copy()
    # if highly_variable_genes == True:
    #     sc.pp.highly_variable_genes(adata,
    #                                 flavor='seurat_v3',
    #                                 n_top_genes=highly_variable_gene_num or 3000,
    #                                 )
    #     adata = adata[:, adata.var.highly_variable]

    if regress_out == True:
        mito_genes = adata.var_names.str.startswith('MT-')
        adata.obs['percent_mito'] = np.sum(adata[:, mito_genes].X, axis=1) / np.sum(adata.X, axis=1)
        sc.pp.filter_cells(adata, min_counts=0)
        sc.pp.regress_out(adata, ['n_counts', 'percent_mito'])

    if scale == True:
        sc.pp.scale(adata, max_value=scale_max_value, zero_center=scale_zero_center)

    return adata

def find_mutual_nn(data1,
                   data2,
                   dist_method,
                   k1,
                   k2,
                  ):
    if dist_method == 'cosine':
        cos_sim1 = cosine_similarity(data1, data2)
        cos_sim2 = cosine_similarity(data2, data1)
        k_index_1 = torch.topk(torch.tensor(cos_sim2), k=k2, dim=1)[1]
        k_index_2 = torch.topk(torch.tensor(cos_sim1), k=k1, dim=1)[1]
    else:
        dist = DistanceMetric.get_metric(dist_method)
        k_index_1 = KDTree(data1, metric=dist).query(data2, k=k2, return_distance=False)
        k_index_2 = KDTree(data2, metric=dist).query(data1, k=k1, return_distance=False)
    mutual_1 = []
    mutual_2 = []
    mutual = []
    for index_2 in range(data2.shape[0]):
        for index_1 in k_index_1[index_2]:
            if index_2 in k_index_2[index_1]:
                mutual_1.append(index_1)
                mutual_2.append(index_2)
                mutual.append([index_1, index_2])
    return mutual


def intra_exp_adj(adata,
                  find_neighbor_method='KNN',
                  dist_method='euclidean',
                  PCA_dimensionality_reduction=True,
                  dim=50,
                  corr_dist_neighbors=10,
                  ):
    ST_exp = adata.copy()

    sc.pp.scale(ST_exp, max_value=None, zero_center=True)
    if PCA_dimensionality_reduction == True:
        sc.tl.pca(ST_exp, n_comps=dim, svd_solver='arpack', random_state=None)
        input_data = ST_exp.obsm['X_pca']
        if find_neighbor_method == 'KNN':
            if dist_method == 'cosine':
                cos_sim = cosine_similarity(input_data, input_data)
                k_index = torch.topk(torch.tensor(cos_sim), k=corr_dist_neighbors, dim=1)[1]
            else:
                dist = DistanceMetric.get_metric(dist_method)
                k_index = KDTree(input_data, metric=dist).query(input_data, k=corr_dist_neighbors,
                                                                return_distance=False)
            A_exp = np.zeros((ST_exp.shape[0], ST_exp.shape[0]), dtype=float)
            for i in range(k_index.shape[0]):
                for j in k_index[i]:
                    if i != j:
                        A_exp[i, j] = 1;
                        A_exp[j, i] = 1;
            A_exp = pd.DataFrame(A_exp, index=ST_exp.obs.index.values, columns=ST_exp.obs.index.values)
        elif find_neighbor_method == 'MNN':
            mut = find_mutual_nn(input_data, input_data, dist_method=dist_method, k1=corr_dist_neighbors,
                                 k2=corr_dist_neighbors)
            mut = pd.DataFrame(mut, columns=['data1', 'data2'])
            A_exp = np.zeros((ST_exp.shape[0], ST_exp.shape[0]), dtype=float)
            for i in mut.index:
                A_exp[mut.loc[i, 'data1'], mut.loc[i, 'data2']] = 1
                A_exp[mut.loc[i, 'data2'], mut.loc[i, 'data1']] = 1
            A_exp = A_exp - np.eye(A_exp.shape[0])
            A_exp = pd.DataFrame(A_exp, index=ST_exp.obs.index.values, columns=ST_exp.obs.index.values)
    else:
        sc.pp.scale(ST_exp, max_value=None, zero_center=True)
        input_data = ST_exp.X
        if find_neighbor_method == 'KNN':
            if dist_method == 'cosine':
                cos_sim = cosine_similarity(input_data, input_data)
                k_index = torch.topk(torch.tensor(cos_sim), k=corr_dist_neighbors, dim=1)[1]
            else:
                dist = DistanceMetric.get_metric(dist_method)
                k_index = KDTree(input_data, metric=dist).query(input_data, k=corr_dist_neighbors,
                                                                return_distance=False)
            A_exp = np.zeros((ST_exp.shape[0], ST_exp.shape[0]), dtype=float)
            for i in range(k_index.shape[0]):
                for j in k_index[i]:
                    if i != j:
                        A_exp[i, j] = 1;
                        A_exp[j, i] = 1;
            A_exp = pd.DataFrame(A_exp, index=ST_exp.obs.index.values, columns=ST_exp.obs.index.values)
        elif find_neighbor_method == 'MNN':
            nan_mask = np.isnan(input_data)
            row_indices, col_indices = np.where(nan_mask)
            if col_indices.size > 0:
                # print("fillna")
                input_data[row_indices, col_indices] = 0

            mut = find_mutual_nn(input_data, input_data, dist_method=dist_method, k1=corr_dist_neighbors,
                                 k2=corr_dist_neighbors)
            mut = pd.DataFrame(mut, columns=['data1', 'data2'])
            A_exp = np.zeros((ST_exp.shape[0], ST_exp.shape[0]), dtype=float)
            for i in mut.index:
                A_exp[mut.loc[i, 'data1'], mut.loc[i, 'data2']] = 1
                A_exp[mut.loc[i, 'data2'], mut.loc[i, 'data1']] = 1
            A_exp = A_exp - np.eye(A_exp.shape[0])
            A_exp = pd.DataFrame(A_exp, index=ST_exp.obs.index.values, columns=ST_exp.obs.index.values)

    return A_exp



# def build_edge(input_tensor, method, k):
#     knn = NearestNeighbors(n_neighbors=k, metric="cosine")
#     knn.fit(input_tensor)
#     indices = knn.kneighbors(input_tensor, return_distance=False)
#     edges = torch.zeros((len(input_tensor), len(input_tensor)))
#     if method == "knn":
#         for i in range(len(input_tensor)):
#             for j in indices[i]:
#                 edges[i, j] = 1
#     elif method == "mnn":
#         mnn = [[] for j in range(len(input_tensor))]
#         for i in range(len(input_tensor)):
#             # Get the indices of the nearest neighbors for the current data point
#             neighbors = indices[i]
#             # Iterate over each nearest neighbor
#             for j in neighbors:
#                 # Check if the current data point is also a nearest neighbor of its neighbor
#                 if i in indices[j]:
#                     # Set the corresponding entry in the MNN matrix to 1
#                     edges[i, j] = 1
#                     mnn[i].append(j)
#         indices = mnn
#     return edges, indices


if __name__ == "__main__":
    train_data_path = "data/starmap/pseudo_ST.h5ad"
    train_data = sc.read_h5ad(train_data_path)
    train_ex = intra_exp_adj(train_data[:200], dist_method="cosine", corr_dist_neighbors=6,
                             PCA_dimensionality_reduction=False,
                             find_neighbor_method='MNN')
    print(train_ex.shape)