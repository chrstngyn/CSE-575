from Precode import *
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd

class KMeans1(object):
    def __init__(self, k: int, points: list, data: list):
        self.clusters = None
        self.data = data
        self.k = k
        self.loss = None
        self.points = points

    # k: the number of clusters
    # points:  2nd list of randomly generated mu values
    # data: 2d list of data points to be classified
    def calc_k_means(self):
        changed = True
        while changed:
            self.clusters = {}

            for i in range(1, self.k + 1):
                self.clusters[i] = []

            for v in data:
                distance = float('inf')
                current_cluster = 0

                for i in range(len(self.points)):
                    dist = np.linalg.norm(self.points[i] - v)
                    if dist < distance:
                        distance = dist
                        current_cluster = i + 1
                self.clusters[current_cluster].append(v.tolist())

            # move mu for each cluster
            for i in range(len(self.points)):
                mu = np.mean(np.array(self.clusters[i + 1]), axis=0)
                if np.array_equal(mu, self.points[i]):
                    changed = False
                else:
                    changed = True
                    self.points[i] = mu


    def calc_object_function(self):
        summation = 0
        for i in range(self.k):
            for j in range(len(self.clusters[i+1])):
                summation += np.linalg.norm(self.clusters[i+1][j] - self.points[i] ** 2)
        self.loss = summation
    
    def plot(self):
        for k in KMS.clusters:
            for point in KMS.clusters[k]:
                plt.plot(point[0], point[1], 'o', color = colors[k-1], label="Cluster='{0}'".format(k))
            plt.plot(KMS.points[k-1][0], KMS.points[k-1][1], 'o', markersize=15, markeredgewidth=2.0, mec= 'k', color=colors[k-1], label="Cluster='{0}'".format(k))
        plt.show()

    def display_output(self):
        print('After KMeans Algorithm: \n', pd.DataFrame(KMS.points, columns=["X1", "X2"]), '\n')
        print('Loss: ', KMS.loss)

class KMeans2(object):
    def __init__(self,  k: int,  points: list, data: list):
        self.clusters = None
        self.data = data
        self.k = k
        self.loss = None
        self.points = [points]

    def add_point(self, indices):
        max_distance = 0

        for i in range(len(self.data)):
            if i in indices.keys():
                continue
            running_distance = 0
        
            for j in range(len(self.points)):
                running_distance += np.linalg.norm(abs(self.points[j] - self.data[i]))

            if running_distance > max_distance:
                max_distance = running_distance
                idx = i
        
        indices[idx] = 1
        self.points.append(self.data[idx])
    
    def calc_initial_centers(self):
        indices = {}
        for i in range(self.k - 1):
            self.add_point(indices)

    def calc_k_means(self):
        changed = True
        while changed:
            self.clusters = {}

            for i in range(1, self.k + 1):
                self.clusters[i] = []

            for v in data:
                distance = float('inf')
                current_cluster = 0

                for i in range(len(self.points)):
                    dist = np.linalg.norm(self.points[i] - v)
                    if dist < distance:
                        distance = dist
                        current_cluster = i + 1
                self.clusters[current_cluster].append(v.tolist())

            # move mu for each cluster
            for i in range(len(self.points)):
                mu = np.mean(np.array(self.clusters[i + 1]), axis=0)
                if np.array_equal(mu, self.points[i]):
                    changed = False
                else:
                    changed = True
                    self.points[i] = mu

    def calc_object_function(self):
        summation = 0
        for i in range(self.k):
            for j in range(len(self.clusters[i+1])):
                summation += np.linalg.norm(np.array(self.clusters[i+1][j]) - np.array(self.points[i])) ** 2
        self.loss = summation

    def plot(self): 
        for k in KMS.clusters:
            for point in KMS.clusters[k]:
                plt.plot(point[0],point[1], 'o', color=colors[k-1], label="Cluster='{0}'".format(k) )
            plt.plot(KMS.points[k-1][0],KMS.points[k-1][1], 'o', markersize=15, markeredgewidth=2.0, mec= 'k', color=colors[k-1], label="Cluster='{0}'".format(k) )
        plt.show()

    def display_output(self):
        print('After KMeans Algorithm: \n', pd.DataFrame(KMS.points, columns=["X", "Y"]), '\n')
        print('Loss: ', KMS.loss)

class Points1(object):
    def get_random_points(k: int, data: list):
        indices = np.random.choice(data.shape[0], k, replace=False)
        return data[indices]

class Points2(object):
    def get_random_points(k: int, data: list):
        indices = np.random.choice(data.shape[0], 1, replace=False)
        return data[indices]

##############

# project strategy 1

data = np.load('AllSamples.npy')

k1, i_point1, k2, i_point2 = initial_S1('4687') # please replace 0111 with your last four digit of your ID

print(k1)
print(i_point1)
print(k2)
print(i_point2)

plt.style.use('ggplot')
colors = ['royalblue', 'mediumpurple', 'navy', 'slateblue', 'rebeccapurple', 'orchid', 'plum', 'mediumvioletred', 'deeppink', 'palevioletred', 'cadetblue', 'dodgerblue', 'mediumseagreen', 'mediumturquoise']

KMS = KMeans1(k1, copy.copy(i_point1), data)
KMS.calc_k_means()
KMS.calc_object_function()
KMS.display_output()
KMS.plot()

kms_loss = []
for k in range(2, 11):
    KMS = KMeans1(k, Points1.getRandomPoints(k, data), data)
    KMS.calc_k_means()
    KMS.calc_object_function()
    kms_loss.append(KMS.loss)
    plt.plot(k, KMS.loss, 'o', markersize=15, markeredgewidth=2.0, mec= 'k', color='b', label="Cluster='{0}'".format(k))
    plt.xlabel("# Clusters")
    plt.ylabel("Loss Value")
    plt.title("Figure 1")

############

# project strategy 2

data = np.load('AllSamples.npy')

k1, i_point1, k2, i_point2 = initial_S2('4687') # please replace 0111 with your last four digit of your ID

KMS = KMeans2(k1, i_point1, data)
KMS.calc_initial_centers()
KMS.calc_k_means()
KMS.calc_object_function()
KMS.display_output()
KMS.plot()

for k in range(2, 11):
    KMS = KMeans2(k, Points2.get_random_points(1, data), data)
    KMS.calc_k_means()
    KMS.calc_object_function()
    kms_loss.append(KMS.loss)
    plt.plot(k, KMS.loss, 'o', markersize=15, markeredgewidth=2.0, mec= 'k', color='b', label="Cluster='{0}'".format(k))
    plt.xlabel("# Clusters")
    plt.ylabel("Loss Value")
    plt.title("k = 4")

KMS = KMeans2(k2, i_point2, data)
KMS.calc_initial_centers()
KMS.calc_k_means()
KMS.calc_object_function()
KMS.display_output()
KMS.plot()

for k in range(2, 11):
    KMS = KMeans2(k, Points2.get_random_points(1, data), data)
    KMS.calc_k_means()
    KMS.calc_object_function()
    kms_loss.append(KMS.loss)
    plt.plot(k, KMS.loss, 'o', markersize=15, markeredgewidth=2.0, mec= 'k', color='b', label="Cluster='{0}'".format(k))
    plt.xlabel("# Clusters")
    plt.ylabel("Loss Value")
    plt.title("k = 6")


for k in range(2, 11):
    KMS = KMeans2(k, Points2.get_random_points(1, data), data)
    KMS.calc_k_means()
    KMS.calc_object_function()
    kms_loss.append(KMS.loss)
    plt.plot(k, KMS.loss, 'o', markersize=15, markeredgewidth=2.0, mec= 'k', color='b', label="Cluster='{0}'".format(k))
    plt.xlabel("# Clusters")
    plt.ylabel("Loss Value")
    plt.title("k = 4")
