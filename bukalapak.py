import numpy as np 
from sklearn.preprocessing import Normalizer
import os
import matplotlib.pyplot as plt
import datetime
from initial_data import generate_all_random_users, generate_all_article_pool
from cd import sclub_cd 
import sys

# Initialization

time_run = datetime.datetime.now().strftime('_%d_%m - %H_%M_%S') 

store_file = 'D:/OneDrive - Institut Teknologi Bandung/Tugas Akhir/Kode/Bukalapak Preprocessed/'
results_folder = 'D:/OneDrive - Institut Teknologi Bandung/Tugas Akhir/Kode/Output/Bukalapak/10_1000/'

user_json = np.load(store_file + 'bukalapak_user_json.npy', allow_pickle = True).item()
article_features = Normalizer().fit_transform(np.load(store_file + 'bukalapak_item_feature.npy', allow_pickle = True))
popular_article_list = np.load(store_file + 'bukalapak_popular_item.npy', allow_pickle = True)

real_data = True
binary_reward = True
dimension = 25
user_num = len(user_json.keys())
article_num = len(article_features)
pool_size = 25
reward_noise_scale = 0.0
top_n_similarity = 300
alpha = 0.05
k = int(sys.argv[1])
iterations = int(sys.argv[2])

all_random_users = generate_all_random_users(iterations, user_json)
all_article_pool = generate_all_article_pool(iterations, all_random_users, user_json, pool_size, article_num, popular_article_list)

# Model and Run

sclub_cd_gnmc = sclub_cd(user_num, article_num, pool_size, dimension, user_json, article_features, alpha, top_n_similarity = top_n_similarity, real_data = real_data, binary_reward = binary_reward, method = 'gnmc', k=k)

start = datetime.datetime.now()

gnmc_bad_cluster, gnmc_cum_regret, gnmc_n_cluster, gnmc_clusters, gnmc_affinity_matrix, gnmc_cum_reward, gnmc_diff_user_features, gnmc_diff_user_cluster_features, gnmc_clustering_score, gnmc_users_served_items, gnmc_served_users = sclub_cd_gnmc.run(iterations, reward_noise_scale, all_random_users, all_article_pool, real_clusters = None)

duration_gnmc = datetime.datetime.now() - start

print('GNMC Duration ', duration_gnmc)

sclub_cd_louvain = sclub_cd(user_num, article_num, pool_size, dimension, user_json, article_features, alpha, top_n_similarity = top_n_similarity, real_data = real_data, binary_reward = binary_reward, method = 'louvain', k=k)

start = datetime.datetime.now()

louvain_bad_cluster, louvain_cum_regret, louvain_n_cluster, louvain_clusters, louvain_affinity_matrix, louvain_cum_reward, louvain_diff_user_features, louvain_diff_user_cluster_features, louvain_clustering_score, louvain_users_served_items, louvain_served_users = sclub_cd_louvain.run(iterations, reward_noise_scale, all_random_users, all_article_pool, real_clusters = None)

duration_louvain = datetime.datetime.now() - start

print('Louvain Duration ', duration_louvain)

# Saving

newpath = results_folder + 'results_' + str(time_run) + '/'
if not os.path.exists(newpath):
    os.makedirs(newpath)

np.save(newpath + 'louvain_cum_regret', louvain_cum_regret)
np.save(newpath + 'louvain_cum_reward', louvain_cum_reward)
np.save(newpath + 'louvain_n_cluster', louvain_n_cluster)
np.save(newpath + 'louvain_clusters', louvain_clusters)
np.save(newpath + 'louvain_similarity_matrix', louvain_affinity_matrix)
np.save(newpath + 'louvain_diff_user_features', louvain_diff_user_features)
np.save(newpath + 'louvain_bad_cluster', louvain_bad_cluster)
np.save(newpath + 'louvain_duration', duration_louvain)

np.save(newpath + 'gnmc_cum_regret', gnmc_cum_regret)
np.save(newpath + 'gnmc_cum_reward', gnmc_cum_reward)
np.save(newpath + 'gnmc_n_cluster', gnmc_n_cluster)
np.save(newpath + 'gnmc_clusters', gnmc_clusters)
np.save(newpath + 'gnmc_similarity_matrix', gnmc_affinity_matrix)
np.save(newpath + 'gnmc_diff_user_features', gnmc_diff_user_features)
np.save(newpath + 'gnmc_bad_cluster', gnmc_bad_cluster)
np.save(newpath + 'gnmc_duration', duration_gnmc)

# Plotting

plt.figure(figsize = (5,5))
plt.plot(louvain_cum_regret, label = 'SCLUB-CD + Louvain',color = 'r', marker = 'X', linewidth = 1, markevery = 0.1, markersize = 8)
plt.plot(gnmc_cum_regret, label = 'SCLUB-CD + GNMC',color = 'b', marker = 'o', linewidth = 1, markevery = 0.1, markersize = 8)
plt.legend(loc = 'best', fontsize = 12)
plt.yticks(rotation = 90)
plt.xlabel('Time', fontsize = 12)
plt.ylabel('Cumulative Regret', fontsize = 12)
plt.ticklabel_format(axis = "both", style = "sci", scilimits = (0,0))
plt.savefig(newpath + 'Cumulative Regret' + '.png', dpi = 600)
plt.show()

plt.figure(figsize = (5,5))
plt.plot(louvain_cum_reward, label = 'SCLUB-CD + Louvain',color = 'r', marker = 'X', linewidth = 1, markevery = 0.1, markersize = 8)
plt.plot(gnmc_cum_reward, label = 'SCLUB-CD + GNMC',color = 'b', marker = 'o', linewidth = 1, markevery = 0.1, markersize = 8)
plt.legend(loc = 'best', fontsize = 12)
plt.yticks(rotation = 90)
plt.ylabel('Cumulative Reward', fontsize = 12)
plt.ticklabel_format(axis = "both", style = "sci", scilimits = (0,0))
plt.savefig(newpath + 'Cumulative Reward' + '.png', dpi = 600)
plt.show()

plt.figure(figsize = (5,5))
plt.plot(louvain_n_cluster, label = 'SCLUB-CD + Louvain',color = 'r', marker = 'X', linewidth = 1, markevery = 0.1, markersize = 8)
plt.plot(gnmc_n_cluster, label = 'SCLUB-CD + GNMC',color = 'b', marker = 'o', linewidth = 1, markevery = 0.1, markersize = 8)
plt.legend(loc = 'best',fontsize = 12)
# plt.ylim([0,100])
plt.yticks(rotation = 90)
plt.ylabel('Cluster Number', fontsize = 12)
plt.xlabel('Time', fontsize = 12)
plt.ticklabel_format(axis = "both", style = "sci", scilimits = (0,0))
plt.savefig(newpath + 'Cluster Number' + '.png', dpi = 600)
plt.show()

plt.figure(figsize = (5,5))
plt.plot(louvain_bad_cluster, label = 'SCLUB-CD + Louvain',color = 'r', marker = 'X', linewidth = 1, markevery = 0.1, markersize = 8)
plt.plot(gnmc_bad_cluster, label = 'SCLUB-CD + GNMC',color = 'b', marker = 'o', linewidth = 1, markevery = 0.1, markersize = 8)
plt.legend(loc = 'best',fontsize = 12)
plt.yticks(rotation = 90)
plt.ylabel('Badly Connected Cluster Number', fontsize = 12)
plt.xlabel('Time', fontsize = 12)
plt.ticklabel_format(axis = "both", style = "sci", scilimits = (0,0))
plt.savefig(newpath + 'Badly Connected Cluster Number' + '.png', dpi = 600)
plt.show()