# clustering people based on quiz answers

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from mpl_toolkits.mplot3d import Axes3D
import plotly.express as px

def load_data(file_path):
    df = pd.read_csv(file_path, header=None)
    data = df.values
    names = data[:, 0]
    coordinates = data[:, 1:].astype(float)
    return names, coordinates

def reduce_dimensions(data, n=3):
    if data.shape[1] <= n:
        return data
    print(data.shape[0])
    mean = np.mean(data, axis=0)
    #print(mean,end="\n\n")
    centeredata = data - mean
    #print(centered_data)

    covmatrix = np.zeros((data.shape[1], data.shape[1]))
    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            cov_ij = np.sum(centeredata[:, i] * centeredata[:, j]) / data.shape[1]
            covmatrix[i, j] = cov_ij

    #covmatrix=np.cov(centeredata, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covmatrix)
    sorted_indices = np.argsort(eigenvalues)
    d_sorted_indices = sorted_indices[::-1]
    sortedevectors = eigenvectors[:, d_sorted_indices]
    topevectors = sortedevectors[:, :n]

    reducedata = centeredata @ topevectors
    return reducedata

def initialize_centroids(data, k):
    indices = np.random.choice(data.shape[0], size=k, replace=False)
    #print(indices,data[indices],sep="\n\n")
    return data[indices]

def assign_clusters(data, centroids):
    clusters = []
    for point in data:
        distances = np.zeros(centroids.shape[0])
        for i in range(centroids.shape[0]):
            sqdiff = (point - centroids[i])**2
            sumsqdiff = np.sum(sqdiff)
            distances[i] = np.sqrt(sumsqdiff)
        cluster = np.argmin(distances)
        clusters.append(cluster)

    return clusters

def update_centroids(data, clusters, k):
    new_centroids = np.zeros((k, data.shape[1]))
    for i in range(k):
        points_in_cluster = data[np.array(clusters) == i]
        if len(points_in_cluster) > 0:
            meanvals = np.mean(points_in_cluster, axis=0)
            new_centroids[i] = meanvals
    return new_centroids

def k_means_clustering(data, k, max_iterations=100):
    centroids = initialize_centroids(data, k)

    for _ in range(max_iterations):
        clusters = assign_clusters(data, centroids)
        new_centroids = update_centroids(data, clusters, k)

        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return clusters,centroids

def detect_anomalies(data, cluster_labels, centroids, threshold):
    anomalies = []
    for i, point in enumerate(data):
        cluster_idx = cluster_labels[i]
        distance = np.linalg.norm(point - centroids[cluster_idx])
        if distance > threshold:
            anomalies.append((i, point))
    return anomalies

def display_clusters(names, cluster_labels):
    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(names[i])
    print("\nPeople Cluster-wise:")
    for cluster, people in clusters.items():
        print(f"\n  Cluster {cluster + 1}")
        for person in people:
            print(f"    {person}")

def display_anomalies(names, anomalies, cluster_labels):
    print("\nAnomalies:")
    for idx, point in anomalies:
        cluster = cluster_labels[idx]  # Get the cluster for the anomaly
        print(f"  {names[idx]} - Cluster {cluster + 1}")

def graph_clusters(names, reduced_data, cluster_labels):
    num_dimensions = reduced_data.shape[1]

    if num_dimensions == 1:
        fig = px.scatter(x=reduced_data.flatten(), y=[0]*len(reduced_data), color=cluster_labels, hover_name=names)
        fig.update_layout(title='K-Means Clustering Visualization (1D)',
                          xaxis_title='Dimension 1',
                          yaxis_title='Value',
                          yaxis=dict(showticklabels=False))
        fig.show()

    elif num_dimensions == 2:
        fig = px.scatter(x=reduced_data[:, 0], y=reduced_data[:, 1], color=cluster_labels, hover_name=names)
        fig.update_layout(title='K-Means Clustering Visualization (2D)',
                          xaxis_title='Dimension 1',
                          yaxis_title='Dimension 2')
        fig.show()

    elif num_dimensions == 3:
        fig = px.scatter_3d(x=reduced_data[:, 0], y=reduced_data[:, 1], z=reduced_data[:, 2], color=cluster_labels, hover_name=names)
        fig.update_layout(title='K-Means Clustering Visualization (3D)',
                          scene=dict(
                              xaxis_title='Dimension 1',
                              yaxis_title='Dimension 2',
                              zaxis_title='Dimension 3'))
        fig.show()

file_path = input("Enter the path to the CSV file: ")
n=int(input("Enter how many dimensions you want to reduce it to (1-3): "))
while(n<1 or n>3):
    n=int(input("Please enter a value from 1-3: "))
num_clusters = int(input("Enter the number of clusters: "))
names, coordinates = load_data(file_path)

#print(coordinates)
reduced_data = reduce_dimensions(coordinates, n)
cluster_labels, centroids = k_means_clustering(reduced_data, num_clusters)
display_clusters(names, cluster_labels)
anomalies = detect_anomalies(reduced_data, cluster_labels, centroids,7)
display_anomalies(names, anomalies, cluster_labels)
graph_clusters(names, reduced_data, cluster_labels)
