import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csc_matrix, coo_matrix
from sklearn.cluster import KMeans
import pandas as pd
import warnings

warnings.filterwarnings('ignore')



def node_id_map(nodes_used):
    '''
    :param nodes_used: list of nodes
    :return: mapping between node_name and id
            mapping bewteen id and node name
    '''
    node2id = {}
    for i, node in enumerate(nodes_used):
        node2id[node] = i

    id2node = {v: k for k, v in node2id.items()}

    return node2id, id2node


def pre_process(nodes_file_path, edges_file_path):
    '''

    :param nodes_file_path: nodes
    :param edges_file_path: edges
    :return: pandas df with 'used' nodes and pandas df with edges
    '''
    df_nodes = pd.read_csv(nodes_file_path, sep='\t', header=None, index_col=0, names=['id', 'name', 'label', 'type'])
    df_edges = pd.read_csv(edges_file_path, sep='\t', header=None, names=['from', 'to'])

    nodes_used = np.unique(np.concatenate([df_edges['from'], df_edges['to']], axis=None))
    df_nodes_filtered = df_nodes.copy().iloc[nodes_used - 1, :]

    node2id, id2node = node_id_map(nodes_used)

    df_edges['from_id'] = df_edges['from'].apply(lambda x: node2id[x])
    df_edges['to_id'] = df_edges['to'].apply(lambda x: node2id[x])

    return df_nodes_filtered, df_edges, node2id, id2node

def compute_adjacency_matrix(df_nodes, df_edges):
    '''

    :param df_nodes: nodes
    :param df_edges: edges
    :return: adjacency matrix
    '''
    i = df_edges['from_id']
    j = df_edges['to_id']
    v = np.ones((df_edges.shape[0], 1)).flatten()

    n = len(df_nodes)
    A = coo_matrix((v, (i, j)), shape=(n, n))
    A = (A + np.transpose(A))
    A = csc_matrix.todense(A)  # ## convert to dense matrix

    return A

def compute_spectrum(A):
    '''

    :param A: Adjacency matrix
    :return: eigenvalues, eigenvectors of Laplacian
    '''
    D = np.diag(1 / np.sqrt(np.sum(A, axis=1)).A1)
    L = np.identity(D.shape[0]) - D @ A @ D
    L = np.array(L)

    eigenvalues, eigenvectors = np.linalg.eig(L)


    return eigenvalues, eigenvectors

def find_optimal_clusters(eigenvalues, num_trials):
    '''
    :param eigen_values: eigen_values of Laplacian
    :return: best number of clusters based on eigen gap
    '''

    index_largest_gap = np.argsort(np.diff(eigenvalues))[::-1][:num_trials]
    nb_clusters = index_largest_gap + 1

    return nb_clusters

def compute_missmatch_rate(eigenvectors, num_clusters, id2node, df_nodes):
    '''

    :param eigenvectors:
    :param num_clusters:
    :param id2node:
    :param df_nodes:
    :return:
    '''

    z = eigenvectors[:, 0:num_clusters].real

    Kmeans = KMeans(n_clusters=num_clusters).fit(z)
    labels = Kmeans.labels_

    cluster = pd.Series({id2node[i]: cluster_id for i, cluster_id in enumerate(labels)})
    df_nodes['cluster'] = cluster

    majoity = df_nodes.groupby('cluster') \
        .agg({'label': lambda x: pd.Series.mode(x)}) \
        .rename(columns={'label': 'majority_label'}) \
        .reset_index()

    df_final = df_nodes.merge(majoity, left_on='cluster', right_on='cluster')
    df_final['is_mismatch'] = (df_final['label'] != df_final['majority_label']).astype(int)

    mismatch_rates = (df_final.groupby('cluster')['is_mismatch'].sum() / df_final.groupby('cluster')[
        'is_mismatch'].count()) * 100.0

    overall_missmatch_rate = round(df_final['is_mismatch'].sum() * 100.0 / df_final.shape[0], 2)

    return mismatch_rates, overall_missmatch_rate

def SpectralClustering(nodes_file_path, edges_file_path, k=2, optimize=False, plot = False):
    '''
    :param nodes_file_path: file path of nodes file
    :param edges_file_path: fil path for edges file
    :return: labels and mismatch %
    '''

    df_nodes, df_edges, node2id, id2node = pre_process(nodes_file_path, edges_file_path)
    A = compute_adjacency_matrix(df_nodes, df_edges)

    eigenvalues, eigenvectors = compute_spectrum(A)
    idx_sorted = np.argsort(eigenvalues)  # the index of eigenvalue sorted acsending
    eigenvalues_sorted = eigenvalues[idx_sorted]
    eigenvectors_sorted = eigenvectors[:, idx_sorted]

    if optimize:
        nb_clusters = find_optimal_clusters(eigenvalues, 5)
        missmatch_rates_tracker = []
        for i in nb_clusters:
            _, overall_missmatch_rate = compute_missmatch_rate(eigenvectors_sorted, i, id2node, df_nodes)
            missmatch_rates_tracker.append(overall_missmatch_rate)

            print(f'for n_clusters = {i}, the overall missmatch rate = {overall_missmatch_rate}')

        optimal_k_index = np.argmin(missmatch_rates_tracker)
        optimal_k = nb_clusters[optimal_k_index]

        k = optimal_k

        print('optimal num clusters is:- ', k, '\n')
        print('-------------------done---------------------')


    missmatch_rates, overall_missmatch_rate = compute_missmatch_rate(eigenvectors_sorted, k, id2node, df_nodes)

    print('--------------------------------------------')
    print('num clusters:-', k)
    print('overall_missmatch_rate: ', overall_missmatch_rate, '%')
    print('missmatch_rats \n', missmatch_rates)

    if plot:

        plt.figure(0)
        plt.spy(A)
        plt.title('Adjacency Matrix')
        plt.savefig(f'Q3_Adjacency Matrix.png')
        plt.show()

        #plt.figure(1)
        #z = eigenvectors_sorted[:, 0:k].real
        #plt.scatter(z[:, 0], z[:, 1])
        #plt.show()

        plt.figure(2)
        plt.scatter(range(len(eigenvalues_sorted)), eigenvalues_sorted.real)
        plt.savefig(f'Q3_spectrum.png')
        plt.show()



def main():
    dirpath = os.getcwd()
    node_file = dirpath + '\\data\\nodes.txt'
    edge_file = dirpath + '\\data\\edges.txt'


    SpectralClustering(node_file, edge_file, k=2, optimize=False, plot=False)
    SpectralClustering(node_file, edge_file, k=5, optimize=False, plot=False)
    SpectralClustering(node_file, edge_file, k=10, optimize=False, plot=False)
    SpectralClustering(node_file, edge_file, k=25, optimize=False, plot=False)

    SpectralClustering(node_file, edge_file, k=2, optimize=True, plot=True)


if __name__ == "__main__":
    main()


