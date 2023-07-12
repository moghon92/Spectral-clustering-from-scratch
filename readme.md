# Spectral clustering
Is a technique used for clustering data points based on their similarity or dissimilarity. Unlike traditional clustering algorithms that operate in the original data space, spectral clustering leverages the spectral properties of a similarity matrix to partition the data into cohesive clusters.

## The spectral clustering algorithm consists of the following steps:

- Construct a similarity graph: First, a similarity matrix is computed to capture the pairwise similarities or dissimilarities between data points. Common choices for measuring similarity include the Gaussian kernel, Euclidean distance, or cosine similarity. The similarity matrix is typically symmetric and non-negative.

- Graph Laplacian matrix: From the similarity matrix, a graph Laplacian matrix is constructed. The Laplacian matrix encodes the structural information of the similarity graph and provides a way to analyze the connectivity of data points.

- Eigenvalue decomposition: The Laplacian matrix is decomposed into its eigenvectors and eigenvalues. The eigenvectors represent the embedding of data points into a low-dimensional space, while the eigenvalues reflect the corresponding importance or relevance of each eigenvector.

- Clustering: The eigenvectors corresponding to the smallest eigenvalues are used to represent the data points in the low-dimensional space. Traditional clustering algorithms, such as k-means or spectral clustering, can be applied to these embedded vectors to group the data points into clusters.

The key idea behind spectral clustering is that the low-dimensional embedding captures the global structure of the data, enabling effective clustering even in cases where the clusters are non-linearly separable in the original data space. By leveraging the graph Laplacian and spectral decomposition, spectral clustering is particularly useful for identifying clusters with complex shapes or when dealing with high-dimensional data.