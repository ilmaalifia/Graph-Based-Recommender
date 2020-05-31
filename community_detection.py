import networkx as nx 
from community import community_louvain
from networkx.algorithms.community.centrality import girvan_newman
import numpy as np
import pandas as pd
from networkx.algorithms.components import connected_components, number_connected_components
import itertools
import sys
import matplotlib.pyplot as plt
import datetime

def count_inter_edges(graph, cluster_1, cluster_2):
    cluster_1_nodes = list(cluster_1)
    cluster_2_nodes = list(cluster_2)
    temp = []

    for node in cluster_1_nodes:
        for n in graph.neighbors(node):
            if n in cluster_2_nodes:
                temp.append(n)

    return len(temp)

def count_intra_edges(graph, cluster):
    cluster_nodes = list(cluster)
    count = 0
    
    for node in cluster_nodes:
        temp = [n for n in graph.neighbors(node)]
        for i in temp:
            if i not in cluster_nodes:
                temp.remove(i)
        count += len(temp)
    
    return int(count / 2)

def intra_connection_density(graph, cluster):
    n = len(cluster)
    if n > 1:
        return (2 * count_intra_edges(graph, cluster)) / (n * (n - 1))
    elif n == 1 or n == 0:
        return 0

def inter_connection_density(graph, cluster_1, cluster_2):
    n_1 = len(cluster_1)
    n_2 = len(cluster_2)
    
    return count_inter_edges(graph, cluster_1, cluster_2) / (n_1 * n_2)

def coupling_degree(graph, cluster_1, cluster_2):
    return inter_connection_density(graph, cluster_1, cluster_2) / (intra_connection_density(graph, cluster_1) + intra_connection_density(graph, cluster_2) + 1)

def mc_modularity(graph, clusters):
    sum = 0
    i = 0
    n_clusters = len(clusters)
    indexes = itertools.combinations([n for n in range(n_clusters)], 2)
    
    if n_clusters > 1:
        for i in indexes:
            sum += coupling_degree(graph, clusters[i[0]], clusters[i[1]])
        return 1 - ((2 / (n_clusters * (n_clusters - 1))) * sum)
    elif n_clusters == 1:
        return 0

def find_community_louvain(graph):
    start = datetime.datetime.now()

    parts = community_louvain.best_partition(graph)
    values = [parts.get(node) for node in sorted(graph.nodes())]
    clusters = values
    n_clusters = len(np.unique(values))

    duration = datetime.datetime.now() - start

    print('Modularity Optimization Duration ------------------ ', duration)

    return clusters, n_clusters, parts

def find_community_gnmc(graph, k = None):
    comp = girvan_newman(graph)
    max_modularity = 0.0
    max_community = None
    modularity = 0.0
    
    if k != None:
        communities = []
        for community in itertools.islice(comp, k):
            communities.append(tuple(sorted(c) for c in community))
    else:
        communities = comp
    
    start = datetime.datetime.now()

    for community in communities:
        new = mc_modularity(graph, community)
        if abs(new - modularity) < sys.float_info.epsilon:
            break
        else:
            modularity = new
            if modularity >= max_modularity:
                max_modularity = modularity
                max_community = community
    
    duration = datetime.datetime.now() - start

    print('MC Modularity Duration ------------------ ', duration)
    
    parts = {}
    if max_community != None:
        n_clusters = len(max_community)
        clusters = []
            
        if n_clusters != 1:
            idx = 0
            if n_clusters > 1:
                for i in max_community:
                    for j in i:
                        parts[j] = idx
                    idx += 1
            elif n_clusters == 1:
                clusters = [0 for i in range(len(max_community[0]))]

            clusters = [parts.get(node) for node in sorted(graph.nodes())]

        else:
            clusters = [0]
            n_clusters = 1
            parts = {0: 0}

    else:
        n_clusters = len(graph.nodes())
        clusters = [0 for n in range(n_clusters)]
        for i in graph.nodes():
            parts[i] = i

    return clusters, n_clusters, parts

def plot_graph_community(graph, clusters, style = 'spring'):
    if style == 'spring':
        nx.draw_spring(graph, cmap = plt.get_cmap('jet'), node_color = clusters, node_size = 35, with_labels = False)
    elif style == 'spectral':
        nx.draw_spectral(graph, cmap = plt.get_cmap('jet'), node_color = clusters, node_size = 35, with_labels = False)
    elif style == 'random':
        nx.draw_random(graph, cmap = plt.get_cmap('jet'), node_color = clusters, node_size = 35, with_labels = False)
    elif style == 'circular':
        nx.draw_circular(graph, cmap = plt.get_cmap('jet'), node_color = clusters, node_size = 35, with_labels = False)
    elif style == 'shell':
        nx.draw_shell(graph, cmap = plt.get_cmap('jet'), node_color = clusters, node_size = 35, with_labels = False)

def badly_connected(graph, node_cluster):
    # node_cluster = list of node in one cluster
    sub = graph.subgraph(node_cluster)
    return number_connected_components(sub) > 1

def count_badly_connected(graph, parts):
    # parts = {<node>: <cluster>}
    temp = pd.DataFrame.from_dict(parts, orient='index').reset_index().sort_values(by='index').reset_index(drop=True)
    temp.columns = ['node', 'cluster']
    cluster = list(temp['cluster'].unique())
    cnt = 0
    bad_cluster = []
    for i in cluster:
        if badly_connected(graph, list(temp[temp['cluster'] == i]['node'])):
            bad_cluster.append(i)
            cnt += 1
    return cnt