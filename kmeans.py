# John Strenio CS510 CV & DL Assignment1: Part1
# citations: https://matplotlib.org/stable/gallery/mplot3d/scatter3d.html

# K MEANS ALGORITHM ====================
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

if len(sys.argv) != 2:
    print("the image or dataset is provided as a cmd line arg")
    exit()

# get user input
k = int(input("number of clusters: "))
dims = int(input("dataset dimensions: "))
runs = int(input("number of runs: "))

for r in range(runs):
    # vars
    all_clusters = []
    all_centroids = []
    all_sse = []

    def calc_dist(centroid, data_pt):
        return np.linalg.norm(centroid - data_pt)

    def place_in_cluster(data_pt, pos):
        cluster_dists = [0] * k

        # calculate the distances to centroids
        for i in range(k):
            cluster_dists[i] = calc_dist(centroids[i], data_pt)

        # retrieve index and place in nearest cluster
        index = np.argmin(cluster_dists)
        new_clusters[index].append(data_pt)

        # update pixel cluster position
        if dims == 3:
            cluster_mtx[pos[0]][pos[1]] = index

    # read in data
    if dims == 2:
        data = np.genfromtxt(sys.argv[1])
    else:
        # resize for quicker operation
        data = cv2.imread(sys.argv[1])
        img_h, img_w, img_channels = data.shape
        img_h = int(img_h / 2)
        img_w = int(img_w / 2)
        data = cv2.resize(data, (img_w, img_h))
        cluster_mtx = np.zeros((img_h, img_w), np.uint8)

    # vars
    clusters = [[] for i in range(k)]
    centroids = [0] * k
    change = True

    # shuffle data
    if dims == 2:
        np.random.shuffle(data)

    # randomly assign centroids to k data points
    for i in range(k):
        if dims == 3:
            rh = np.random.randint(0, len(data))
            rw = np.random.randint(0, len(data[0]))
            centroids[i] = data[rh][rw]
        else:
            centroids[i] = data[i]

    # while the centroids continue to change
    while(change):
        # create new clusters to fill
        new_clusters = [[] for i in range(k)]

        if dims == 3:
            for i in range(len(data)):
                for j in range(len(data[i])):
                    place_in_cluster(data[i][j], (i, j))
        else:
            # place data in nearest cluster
            for i in range(len(data)):
                place_in_cluster(data[i], i)

        # find new centroid
        cur_changes = False
        new_centroids = []
        for i in range(k):
            new_centroids.append(np.average(new_clusters[i], axis=0))

        # check if the centroid changed
        if np.array_equal(centroids, new_centroids):
            change = False
        
        # update clusters
        clusters = new_clusters
        centroids = new_centroids

    # CALC SSE =================================== 
    sq_errors = []

    # calc distance from each data point to centroid and square
    for i in range(k):
        for j in range(len(clusters[i])):
            sq_errors.append((calc_dist(centroids[i], clusters[i][j]) ** 2))

    # sum them
    sse = sum(sq_errors)

    # track results
    all_sse.append(sse)
    all_clusters.append(clusters)
    all_centroids.append(centroids)

# store best run based on SSE
index_best = all_sse.index(min(all_sse))
best_clusters = all_clusters[index_best]
best_centroids = all_centroids[index_best]

# write out new image
if dims == 3:
    output = np.zeros((img_h, img_w, img_channels), np.uint8)
    for i in range(img_h):
        for j in range(img_w):
            output[i][j] = best_centroids[cluster_mtx[i][j]]
    cv2.imwrite("KMeans_outputimg2.jpg", output)

# PLOTTING ================================
fig = plt.figure()

if (dims == 3):
    ax = Axes3D(fig)

colors = ["red", "blue", "green", "yellow", "orange", "black"]
plot_clusters_dim1 = [[] for i in range(k)]
plot_clusters_dim2 = [[] for i in range(k)]
plot_clusters_dim3 = [[] for i in range(k)]
plot_centroids_dim1 = [[] for i in range(k)]
plot_centroids_dim2 = [[] for i in range(k)]
plot_centroids_dim3 = [[] for i in range(k)]

# for each cluster, for each data point, for each dimension
for i in range(len(best_clusters)):
    for j in range(len(best_clusters[i])):
        # rearrange add to the list of data points
        plot_clusters_dim1[i].append(best_clusters[i][j][0])
        plot_clusters_dim2[i].append(best_clusters[i][j][1])
        if (dims == 3):
            plot_clusters_dim3[i].append(best_clusters[i][j][2])

# add data points to plots
for i in range(k):
    if (dims == 3):
        ax.scatter(plot_clusters_dim1[i], plot_clusters_dim2[i], plot_clusters_dim3[i], color=colors[i])
    else:
        plt.scatter(plot_clusters_dim1[i], plot_clusters_dim2[i], color=colors[i])

# rearrange and add centroid points
for i in range(len(best_centroids)):
    plot_centroids_dim1[i].append(best_centroids[i][0])
    plot_centroids_dim2[i].append(best_centroids[i][1])
    if (dims == 3):
        plot_centroids_dim3[i].append(best_centroids[i][2])

# add centroids to plot lists
if (dims == 3):
    ax.scatter(plot_centroids_dim1, plot_centroids_dim2, plot_centroids_dim2, color=colors[5])
else:
    plt.scatter(plot_centroids_dim1, plot_centroids_dim2, color=colors[5])

print("best SSE run " + str(index_best) + ": " + str(all_sse[index_best]))
plt.show()
