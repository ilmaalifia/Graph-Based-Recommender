import numpy as np
import matplotlib.pyplot as plt
import networkx as nx 

def feature_uniform(dimension):
    vector = np.array([np.random.uniform( ) for _ in range(dimension)])
    l2_norm = np.linalg.norm(vector, ord = 2)
    vector = vector / l2_norm
    return vector

def init_user_features(user_num, user_dimension, random = None):
    if random:
        user_features = np.zeros((user_num, user_dimension))
        for i in range(user_num):
            user_features[i] = feature_uniform(user_dimension)
    else:
        user_features = np.zeros((user_num, user_dimension))
    return user_features

def init_cor_matrix(user_num, dimension):
    all_cor_matrix = np.array([np.identity(dimension) for i in range(user_num)])
    return all_cor_matrix

def init_bias(user_num, dimension):
    all_bias = np.zeros([user_num, dimension])
    return all_bias

def init_user_cluster_features(user_num, dimension):
    user_cluster_feature = np.zeros([user_num, dimension])
    return user_cluster_feature

def generate_graph_from_adj_matrix(adj_matrix):
    adj_matrix = np.matrix(adj_matrix)
    G = nx.from_numpy_matrix(adj_matrix)
    print(nx.info(G))
    return G

def plot_graph(graph, style = 'spring'):
    if style == 'spring':
        nx.draw_spring(graph, with_labels = False, node_size = 10)
    elif style == 'spectral':
        nx.draw_spectral(graph, with_labels = False, node_size = 10)
    elif style == 'random':
        nx.draw_random(graph, with_labels = False, node_size = 10)
    elif style == 'circular':
        nx.draw_circular(graph, with_labels = False, node_size = 10)
    elif style == 'shell':
        nx.draw_shell(graph, with_labels = False, node_size = 10)

def generate_all_random_users(iterations, user_json):
    all_random_users = []
    for i in range(iterations):
        all_random_users.extend(np.random.choice(list(user_json.keys()), 1, replace = True).tolist())
    return all_random_users

def generate_all_article_pool(iterations, all_random_users, user_json, pool_size, article_num, pool):
    all_article_pool = []
    for i in range(iterations):
        selected_user = all_random_users[i]
        article_pool = np.random.choice(pool, pool_size - 1, replace = True).tolist()
        if user_json[selected_user] != []:
            another_article = np.random.choice(user_json[selected_user], 1).tolist()
            article_pool.extend(another_article)
        else:
            pass
        all_article_pool.append(article_pool)
    return all_article_pool