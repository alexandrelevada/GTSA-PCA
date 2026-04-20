#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Curvature-Aware PCA with Geodesic Tangent Space Aggregation for Semi-Supervised Dimensionality Reduction

Python script to reproduce the results depicted in Table 4

@author: Alexandre L. M. Levada

"""

# Imports
import sys
import time
import warnings
import umap
import numpy as np
import scipy as sp
import networkx as nx
import matplotlib.pyplot as plt
import sklearn.datasets as skdata
import sklearn.neighbors as sknn
from sklearn.neighbors import NearestNeighbors
from itertools import combinations
from numpy import log
from numpy import trace
from numpy import dot
from numpy import sqrt
from numpy import exp
from numpy import eye
from numpy.linalg import det
from numpy.linalg import inv
from numpy.linalg import norm
from numpy.linalg import eigvals
from scipy.spatial.distance import cosine
from sklearn import preprocessing
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import HDBSCAN
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from scipy.stats import wasserstein_distance
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import KernelPCA
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import SpectralEmbedding

# To avoid unnecessary warning messages
warnings.simplefilter(action='ignore')

#######################################################################################################################
# Supervised PCA implementation (variation from paper Supervised Principal Component Analysis - Pattern Recognition)
#######################################################################################################################
def SupervisedPCA(dados, labels, d):
    dados = dados.T
    m = dados.shape[0]      # number of samples
    n = dados.shape[1]      # number of features
    I = np.eye(n)
    U = np.ones((n, n))
    H = I - (1/n)*U
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if labels[i] == labels[j]:
                L[i, j] = 1
    Q1 = np.dot(dados, H)
    Q2 = np.dot(H, dados.T)
    Q = np.dot(np.dot(Q1, L), Q2)
    # Eigenvalues and eigenvectors of the covariance matrix
    v, w = np.linalg.eig(Q)
    # Sort the eigenvalues
    ordem = v.argsort()
    # Select the d eigenvectors associated to the d largest eigenvalues
    maiores_autovetores = w[:, ordem[-d:]]
    # Projection matrix
    Wpca = maiores_autovetores
    # Linear projection into the 2D subspace
    novos_dados = np.dot(Wpca.T, dados)
    return novos_dados

#########################################################################################################
# Compute the mean curvature at each point of the k-NN graph (discrete approximation for the manifold)
#########################################################################################################
def Mean_Curvatures(dados, k):
    n = dados.shape[0]
    m = dados.shape[1]
    # First fundamental form
    I = np.zeros((m, m))
    Squared = np.zeros((m, m))
    ncol = (m*(m-1))//2
    Cross = np.zeros((m, ncol))
    # Second fundamental form
    II = np.zeros((m, m))
    S = np.zeros((m, m))
    curvatures = np.zeros(n)
    shapes = np.zeros((n, m, m))
    # Build KNN without dense adjacency
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(dados)
    knn_indices = nbrs.kneighbors(return_distance=False)
    # Main loop
    for i in range(n):       
        indices = knn_indices[i]
        # Covariance matrix
        amostras = dados[indices]
        ni = len(indices)
        if ni > 1:
            I = np.cov(amostras.T)
        else:
            I = np.eye(m)      # to deal with isolated points
        # Eigendecomposition
        v, w = np.linalg.eig(I)
        # Sort eigenvalues
        ordem = v.argsort()
        # Select top eigenvectors
        Wpca = w[:, ordem[::-1]]
        # Computation of the second fundamental form
        for j in range(0, m):
            Squared[:, j] = Wpca[:, j]**2
        col = 0
        for j in range(0, m):
            for l in range(j, m):
                if j != l:
                    Cross[:, col] = Wpca[:, j]*Wpca[:, l]
                    col += 1
        # Add columns of 1's
        Wpca = np.column_stack((np.ones(m), Wpca))
        Wpca = np.hstack((Wpca, Squared))
        Wpca = np.hstack((Wpca, Cross))        
        Q = Wpca
        # Discard first m+1 columns of Q
        H = Q[:, (m+1):]
        II = np.dot(H, H.T)
        S = -np.dot(II, I).real
        curvatures[i] = trace(S)
    return curvatures

####################################################################
# Geodesic Tangent Space aggregation PCA
#####################################################################
def GTSA_PCA(dados, curvatures, nn, h, d):
    n = dados.shape[0]
    m = dados.shape[1]
    Sigma = np.zeros((m, m))
    pcs_matrix = np.zeros((n, m, m))
    # Generate KNN graph
    knnGraph = sknn.kneighbors_graph(dados, n_neighbors=nn, mode='distance', include_self=False)
    A = knnGraph.toarray()
    # Main loop
    for i in range(n):
        vizinhos = A[i, :]
        indices = vizinhos.nonzero()[0]        
        Z = 0
        for j in indices:
            weight = exp(-curvatures[j]/h)            
            Sigma += weight*(np.outer(dados[j, :] - dados[i, :], dados[j, :] - dados[i, :]))
            Z += weight
        Sigma = Sigma/Z
        v, w = np.linalg.eig(Sigma)
        # Sort the eigenvalues
        order = v.argsort()
        # Select the d eigenvectors associated to the d largest eigenvalues
        Wpca = w[:, order[::-1]]     
        # Tangent spaces
        pcs_matrix[i, :, :] = Wpca
    # Computes geodesic distances in G
    G = nx.from_numpy_array(A)
    D = nx.floyd_warshall_numpy(G)        
    # Compute tangent space based weights
    for i in range(n):
        for j in range(n):
            delta = norm(pcs_matrix[i, :, :] - pcs_matrix[j, :, :], ord='fro')
            A[i, j] = (1/sqrt(1 + D[i, j]))*delta            
    # Spectral decomposition
    lambdas, alphas = sp.linalg.eigh(A)
    # Sort eigenvalues and eigenvectors
    indices = lambdas.argsort()[::-1]
    lambdas = lambdas[indices]
    alphas = alphas[:, indices]
    # Select the d largest eigenvectors
    lambdas = lambdas[0:d]
    alphas = alphas[:, 0:d]
    # Computes the intrinsic coordinates
    output = alphas*np.sqrt(lambdas)
    return output

# Optional function to normalize the curvatures to the interval [a, b]
def normalize_curvatures(curv, a, b):
    k = a + (b - a)*(curv - curv.min())/(curv.max() - curv.min())
    return k

# Regular PCA implementation
def myPCA(dados, d):
    # Eigenvalues and eigenvectors of the covariance matrix
    v, w = np.linalg.eig(np.cov(dados.T))
    # Sort the eigenvalues
    ordem = v.argsort()
    # Select the d eigenvectors associated to the d largest eigenvalues
    maiores_autovetores = w[:, ordem[-d:]]
    # Projection matrix
    Wpca = maiores_autovetores
    # Linear projection into the 2D subspace
    novos_dados = np.dot(Wpca.T, dados.T)
    # Trasformed data    
    return novos_dados

'''
Clustering algortihms
'''
def Clustering(dados, target):
    # Number of clusters
    c = len(np.unique(target))
    # Apply clustering algorithm
    clustering = HDBSCAN(min_cluster_size=10).fit(dados)
    # Computes metrics
    ari = metrics.adjusted_rand_score(target, clustering.labels_)
    mi = metrics.normalized_mutual_info_score(target, clustering.labels_)
    fm = metrics.fowlkes_mallows_score(target, clustering.labels_)
    vm = metrics.v_measure_score(target, clustering.labels_)
    return [ari, fm, vm, clustering.labels_]


# Generate scatterplots
def PlotaDados(dados, labels, metodo):
    # Number of classes
    nclass = len(np.unique(labels))
    # Define colors
    if nclass > 11:
        cores = ['black', 'gray', 'rosybrown', 'firebrick', 'red', 'darksalmon', 'sienna', 'sandybrown', 'bisque', 'tan', 'moccasin', 'floralwhite', 'silver', 'gold', 'darkkhaki', 'lightgoldenrodyellow', 'olivedrab', 'chartreuse', 'palegreen', 'darkgreen', 'seagreen', 'mediumspringgreen', 'lightseagreen', 'paleturquoise', 'darkcyan', 'darkturquoise', 'deepskyblue', 'aliceblue', 'slategray', 'royalblue', 'navy', 'blue', 'mediumpurple', 'darkorchid', 'plum', 'm', 'mediumvioletred', 'palevioletred', 'whitesmoke']
        np.random.shuffle(cores)
    else:
        cores = ['blue', 'red', 'cyan', 'black', 'orange', 'magenta', 'green', 'darkkhaki', 'brown', 'purple', 'salmon', 'tan']
    # Make plot
    plt.figure(10)
    for i in range(nclass):
        indices = np.where(labels==i)[0]        
        cor = cores[i]
        plt.scatter(dados[indices, 0], dados[indices, 1], c=cor, marker='.', alpha=0.5)
    # Save file
    nome_arquivo = metodo + '.png'
    plt.title(metodo+' clusters')
    plt.savefig(nome_arquivo)
    plt.close()

#############################################
#              Data loading
#############################################
# HDBSCAN: PCA x KPCA x GTSA-PCA
#X = skdata.load_wine()
#X = skdata.load_breast_cancer()
#X = skdata.fetch_openml(name='Engine1', version=1)
#X = skdata.fetch_openml(name='user-knowledge', version=1)
#X = skdata.fetch_openml(name='mfeat-karhunen', version=1)
X = skdata.fetch_openml(name='mfeat-factors', version=1)
#X = skdata.fetch_openml(name='mfeat-pixel', version=1)
#X = skdata.fetch_openml(name='cardiotocography', version=1)
#X = skdata.fetch_openml(name='pendigits', version=1)
#X = skdata.fetch_openml(name='Fashion-MNIST', version=1)
#X = skdata.fetch_openml(name='UMIST_Faces_Cropped', version=1)
#X = skdata.fetch_openml(name='page-blocks', version=1)
#X = skdata.fetch_openml(name='heart-c', version=1)
#X = skdata.fetch_openml(name='heart-h', version=1)
#X = skdata.fetch_openml(name='AP_Omentum_Kidney', version=1)
#X = skdata.fetch_openml(name='AP_Colon_Kidney', version=1)
#X = skdata.fetch_openml(name='AP_Ovary_Lung', version=1)
#X = skdata.fetch_openml(name='AP_Breast_Ovary', version=1)
#X = skdata.fetch_openml(name='AP_Colon_Lung', version=1)
#X = skdata.fetch_openml(name='AP_Endometrium_Breast', version=1)

dados = X['data']
target = X['target']

# Large datasets are subsampled
if 'details' in X.keys():
    match X['details']['name']:
        case 'mnist_784' | 'Kuzushiji-MNIST' | 'Fashion-MNIST':
            dados, _, target, _ = train_test_split(dados, target, train_size=0.05, random_state=42)       
        case 'isolet':
            dados, _, target, _ = train_test_split(dados, target, train_size=0.25, random_state=42)
        case 'artificial-characters':
            dados, _, target, _ = train_test_split(dados, target, train_size=0.25, random_state=42)
        case 'optdigits':
            dados, _, target, _ = train_test_split(dados, target, train_size=0.5, random_state=42)
        case 'pendigits':
            dados, _, target, _ = train_test_split(dados, target, train_size=0.25, random_state=42)
        case 'letter':
            dados, _, target, _ = train_test_split(dados, target, train_size=0.2, random_state=42)
        case  'page-blocks':
            dados, _, target, _ = train_test_split(dados, target, train_size=0.5, random_state=42)
        case 'USPS':
            dados, _, target, _ = train_test_split(dados, target, train_size=0.25, random_state=42)
        case 'CIFAR_10_small':
            dados, _, target, _ = train_test_split(dados, target, train_size=0.1, random_state=42)

# Number of samples
n = dados.shape[0]
# Number of features
m = dados.shape[1]
# Number of clusters
c = len(np.unique(target))
# Number of neighbors
nn = round(np.log2(n))

print('N = ', n)
print('M = ', m)
print('C = %d' %c)
print('K = %d' %nn)

# Reduce dimensionality to allow curvature computation in high-dimensional data
if m > 50:
    dados = PCA(n_components=min(50, n), random_state=42).fit_transform(dados)
    n, m = dados.shape

# To deal with sparse matrices
if type(dados) == sp.sparse._csr.csr_matrix:
    dados = dados.todense()
    dados = np.asarray(dados)
else:
    # Convert labels to integers
    if not isinstance(dados, np.ndarray):
        cat_cols = dados.select_dtypes(['category']).columns
        dados[cat_cols] = dados[cat_cols].apply(lambda x: x.cat.codes)
        # Convert to numpy
        dados = dados.to_numpy()
le = LabelEncoder()
le.fit(target)
target = le.transform(target)

# Remove nan's
dados = np.nan_to_num(dados)

# Data standardization (to deal with variables having different units/scales)
dados = preprocessing.scale(dados)

# Validation set is used to estimate hyperparameters (gamma or h)
validation, dados, vlabels, target = train_test_split(dados, target, train_size=0.2, random_state=42)

# Number of neighbors
nn_val = round(np.log2(validation.shape[0]))
nn_data = round(np.log2(dados.shape[0]))

# Number of features
d = 2

############# PCA
dados_pca = myPCA(dados, d)

############# Kernel PCA
model = KernelPCA(n_components=d, kernel='rbf')
dados_kpca = model.fit_transform(dados)

################ MC-ISOMAP
print('\nLearning the gamma parameter for GTSA-PCA...')
list_h = [0.5, 1, 5, 10, 100]
K = Mean_Curvatures(validation, nn_val)
K = normalize_curvatures(K, 0, 1)
best, h_star = -1, 0
for i, h in enumerate(list_h):
    inicio = time.time()
    dados_gtsapca = GTSA_PCA(validation, K, nn_val, h, d)
    fim = time.time()
    L_gtsapca = Clustering(dados_gtsapca, vlabels)
    mean = (L_gtsapca[0] + L_gtsapca[1] + L_gtsapca[2])/3.0
    if mean > best:
        best = mean
        h_star = i
print('\n********************************************************')
print('Best h value for GTSA-PCA is: ', list_h[h_star])
print('**********************************************************')

############ GTSA-PCA
inicio = time.time()
K = Mean_Curvatures(dados, nn_data)
K = normalize_curvatures(K, 0, 1)
dados_gtsapca = GTSA_PCA(dados, K, nn_data, list_h[h_star], d)
fim = time.time()

#%%%%%%%%%%% Clustering
L_pca = Clustering(dados_pca.T.real, target)
L_kpca = Clustering(dados_kpca.real, target)
L_gtsapca = Clustering(dados_gtsapca.real, target)

print('\nRegular PCA')
print('ARI = ', L_pca[0])
print('FM = ', L_pca[1])
print('VM = ', L_pca[2])
print()

print('\nKernel PCA')
print('ARI = ', L_kpca[0])
print('FM = ', L_kpca[1])
print('VM = ', L_kpca[2])
print()

print('\nGTSA-PCA')
print('ARI = ', L_gtsapca[0])
print('FM = ', L_gtsapca[1])
print('VM = ', L_gtsapca[2])
print('\nElapsed time = ', (fim - inicio))
print()

# Plot results
if c < 12 and d == 2:
    PlotaDados(dados_pca.T, L_pca[3], 'PCA')
    PlotaDados(dados_kpca, L_kpca[3], 'Kernel PCA')
    PlotaDados(dados_gtsapca, L_gtsapca[3], 'GTSA-PCA')

# For printing each row of Table 4
#print(str(L_pca[0])+'\t'+str(L_pca[1])+'\t'+str(L_pca[2])+'\t'+str(L_kpca[0])+'\t'+str(L_kpca[1])+'\t'+str(L_kpca[2])+'\t'+str(L_gtsapca[0])+'\t'+str(L_gtsapca[1])+'\t'+str(L_gtsapca[2]))