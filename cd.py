from initial_data import init_bias, init_cor_matrix, init_user_cluster_features, init_user_features, generate_graph_from_adj_matrix
from sklearn.metrics import adjusted_rand_score
import numpy as np
from random import choice
from sklearn.metrics.pairwise import rbf_kernel
from community_detection import find_community_gnmc, find_community_louvain, count_badly_connected

class sclub_cd():
    def __init__(self, user_num, article_num, pool_size, dimension, user_json, article_features, alpha, rating = None, top_n_similarity = None, affinity_matrix = None, real_data = False, binary_reward = False, real_reward = False, real_user_features = None, random = None, method = 'louvain', k=None):
        self.user_num = user_num
        self.article_num = article_num 
        self.dimension = dimension 
        self.pool_size = pool_size
        self.alpha = 1 + np.sqrt(np.log(2.0 / alpha) / 2.0)
        self.article_features = article_features
        self.user_json = user_json
        self.real_user_features = real_user_features
        self.random = random
        self.user_features = init_user_features(self.user_num, self.dimension, random = self.random)
        self.cor_matrix = init_cor_matrix(self.user_num, self.dimension)
        self.cluster_cor_matrix = init_cor_matrix(self.user_num, self.dimension)
        self.bias = init_bias(self.user_num, self.dimension)
        self.cluster_bias = init_bias(self.user_num, self.dimension)
        self.user_cluster_features = init_user_cluster_features(self.user_num, self.dimension)
        self.n_cluster = None
        self.clusters = list(range(self.user_num))
        self.affinity_matrix = affinity_matrix
        self.top_n_similarity = top_n_similarity
        self.cluster_size = None
        self.binary_reward = binary_reward
        self.real_data = real_data
        self.rating = rating
        self.real_reward = real_reward
        self.users_served_items = {}
        self.served_users = []
        self.not_served_users = list(range(self.user_num))
        self.bad_cluster_count = 0
        self.method = method
        self.k = k

    def get_optimal_reward(self, selected_user, article_pool):
        if self.real_data:
            if not self.real_reward:
                liked_articles = self.user_json[selected_user]
                max_reward = 0.0    
                common_article = set(article_pool)&set(liked_articles)
                if len(common_article) != 0:
                    max_reward = 1.0
                else:
                    max_reward = 0.0
            else:
                rates = self.rating[selected_user][article_pool]
                max_reward = np.max(rates)        
        else:
            if self.binary_reward:
                rewards = np.dot(self.article_features[article_pool], self.real_user_features[selected_user])
                big_index = np.where(rewards >= 0.0)[0].tolist()
                small_index = np.where(rewards < 0.0)[0].tolist()
                rewards[big_index] = 1.0
                rewards[small_index] = 0.0
                max_reward = np.max(rewards)
            else: 
                rewards = np.dot(self.article_features[article_pool], self.real_user_features[selected_user])
                max_reward = np.max(rewards)

        return max_reward

    def choose_article(self, selected_user, article_pool, time):
        mean = np.dot(self.article_features[article_pool], self.user_cluster_features[selected_user])
        temp1 = np.dot(self.article_features[article_pool], np.linalg.inv(self.cluster_cor_matrix[selected_user]))
        temp2 = np.sum(temp1 * self.article_features[article_pool], axis = 1) * np.log(time + 1)
        idx_article_picked = np.argmax(mean + self.alpha * np.sqrt(temp2))
        article_picked = article_pool[idx_article_picked]
        return article_picked

    def random_choose_article(self, article_pool):
        article_picked = choice(article_pool)
        return article_picked

    def get_reward(self, selected_user, picked_article):
        if self.real_data:
            liked_articles = self.user_json[selected_user]
            reward = 0.0
            if picked_article in liked_articles:
                reward = 1.0
            else:
                reward = 0.0
        else:
            if self.binary_reward:
                reward = np.dot(self.real_user_features[selected_user], self.article_features[picked_article])
                if reward >= 0.0:
                    reward = 1.0
                else:
                    reward = 0.0
            else:
                reward = np.dot(self.real_user_features[selected_user], self.article_features[picked_article])
        return reward

    def get_regret(self, max_reward, reward):
        regret = max_reward - reward
        return regret

    def find_cluster(self, time, iterations, selected_user):
        if (time % 100 != 0):
            pass
        else:           
            if self.affinity_matrix is None:
                self.affinity_matrix = np.zeros([self.user_num, self.user_num])
            else:
                rbf_row = rbf_kernel(self.user_features[selected_user].reshape(1, -1), self.user_features)[0]
                if len(self.not_served_users) == 0:
                    pass 
                else:
                    rbf_row[self.not_served_users] = 0
                big_index = np.argsort(rbf_row)[self.user_num - self.top_n_similarity:]
                small_index = np.argsort(rbf_row)[:self.user_num - self.top_n_similarity]
                rbf_row[small_index] = 0.0
                rbf_row[big_index] = 1.0
                self.affinity_matrix[selected_user,:] = rbf_row
                self.affinity_matrix[:,selected_user] = rbf_row
                del rbf_row, small_index, big_index

            graph = generate_graph_from_adj_matrix(self.affinity_matrix)

            if self.method == 'gnmc':
                self.clusters, self.n_cluster, parts = find_community_gnmc(graph, self.k)
            elif self.method == 'louvain':
                self.clusters, self.n_cluster, parts = find_community_louvain(graph)
            else:
                self.clusters, self.n_cluster, parts = find_community_louvain(graph)
            
            self.bad_cluster_count = count_badly_connected(graph, parts)
            del graph, parts
            print('SCLUB - CD Cluster ~~~~~~~~~~ ', self.n_cluster)

        return self.n_cluster, self.clusters, self.bad_cluster_count

    def update_user_feature(self, selected_user, picked_article, reward):
        self.cor_matrix[selected_user] += np.outer(self.article_features[picked_article], self.article_features[picked_article])
        self.bias[selected_user] += self.article_features[picked_article] * reward
        self.user_features[selected_user] = np.dot(np.linalg.inv(self.cor_matrix[selected_user]), self.bias[selected_user])

    def update_cluster_parameter(self, selected_user, reward, time):
        if (time % 100 != 0):
            pass 
        else:
            same_cluster = np.where(np.array(self.clusters) == self.clusters[selected_user])[0].tolist()
            self.cluster_cor_matrix[selected_user] = np.identity(self.dimension) + np.sum(self.cor_matrix[same_cluster] - np.identity(self.dimension), axis = 0)
            self.cluster_bias[selected_user] = np.sum(self.bias[same_cluster], axis = 0)
            inv_cluster_cor = np.linalg.inv(self.cluster_cor_matrix[selected_user])
            new_cluster_feature = np.dot(inv_cluster_cor, self.cluster_bias[selected_user])
            for i in same_cluster:
                self.user_cluster_features[i] = new_cluster_feature
            del same_cluster, new_cluster_feature

    def run(self, iterations, reward_noise_scale, all_random_users, all_artilce_pool, real_clusters):
        cum_regret = [0]
        cum_reward = [0]
        cum_n_cluster = [0]
        user_features_diff = [0]
        user_cluster_features_diff = [0]
        clustering_score = [0]
        bad_cluster = [0]
        for time in range(iterations):
            print('SCLUB - CD Time: ', time)
            user = all_random_users[time]
            if user in self.served_users:
                pass 
            else:
                self.served_users.extend([user])
                self.users_served_items[user] = []
                self.not_served_users.remove(user)

            article_pool = all_artilce_pool[time]
            optimal_reward = self.get_optimal_reward(user, article_pool)
            n_cluster, clusters, bad_cluster_count = self.find_cluster(time, iterations, user)
            bad_cluster.extend([bad_cluster_count])

            if real_clusters is not None:
                score = adjusted_rand_score(real_clusters, clusters)
                clustering_score.extend([score])
            else:
                pass

            picked_article = self.choose_article(user, article_pool, time)
            reward = self.get_reward(user, picked_article)
            
            # print('user: ', user)
            # print('item: ', picked_article)

            if reward_noise_scale == 0.0:
                noise_reward = reward
            else:
                noise_reward = reward + np.random.normal(loc = 0.0, scale = reward_noise_scale)         
            
            regret = self.get_regret(optimal_reward, reward)

            if picked_article in self.users_served_items[user]:
                pass 
            else:
                self.users_served_items[user].extend([picked_article])
                self.update_user_feature(user, picked_article, noise_reward)

            self.update_cluster_parameter(user, noise_reward,time)

            if self.real_user_features is not None:
                diff_real_and_learned_user_features = np.sum(np.linalg.norm(self.user_features - self.real_user_features, axis = 1))
                user_features_diff.extend([diff_real_and_learned_user_features])
                diff_real_and_learned_user_cluster_features = np.sum(np.linalg.norm(self.user_cluster_features - self.real_user_features, axis = 1))
                user_cluster_features_diff.extend([diff_real_and_learned_user_cluster_features])
                del diff_real_and_learned_user_features, diff_real_and_learned_user_cluster_features
            else:
                pass 

            cum_n_cluster.extend([n_cluster])
            cum_regret.extend([cum_regret[-1] + regret])
            cum_reward.extend([cum_reward[-1] + reward])

        return bad_cluster, np.array(cum_regret), cum_n_cluster, clusters, self.affinity_matrix, np.array(cum_reward), user_features_diff, user_cluster_features_diff, clustering_score, self.users_served_items, self.served_users