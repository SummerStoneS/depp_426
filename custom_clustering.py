import numpy as np
from numba import jitclass, int32
from common import L1 as distance, Location
from progressbar import ProgressBar, Bar, ETA, Percentage
import pandas as pd
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import pickle


class Cluster:
    """
    一个类，包含地点、中心
    """
    def __init__(self, points):
        self.points = points
        self.core = Location(*np.median(points, 0))  # 中心是所有点的坐标的中位数,tuple
        self.n = len(self.points)

    def merge(self, other):
        """
        将两个类合并成第三个类
        """
        return Cluster(self.points+other.points)


def triu(n):
    """
    生成n阶方阵的上三角的下标
    """
    for i in range(n):
        for j in range(i+1, n):
            yield i, j


class EvenCluster:
    """
    聚类，使得每个类的数量尽量均等
    """
    def __init__(self, num_clusters, relative_weight=1.0):
        """
        Parameters
        ==========
        num_clusters: int
            聚类的数量
        relatice_weight: float
            类的大小相对类间距离的权重
        """
        self.num_clusters = num_clusters
        self.relative_weight = relative_weight

    def fit(self, points):
        """
        开始聚类

        Parameters
        ==========
        points: List[Location]
            点的列表
        """
        clusters = [Cluster([point]) for point in points]         # 一开始每个点都是一个独立的类
        n_clusters = len(clusters)
        distances = np.zeros((n_clusters, n_clusters))            # 计算类之间两两的距离
        
        # 计算类之间的平均距离
        avg_distance = 0
        for i, j in triu(n_clusters):
            distances[i, j] = distance(clusters[i].core, clusters[j].core)
            avg_distance += distances[i, j]
        avg_distance /= n_clusters * (n_clusters - 1) / 2

        # 类的平均大小
        avg_cluster_size = len(points) / self.num_clusters

        # 两两类分组，根据每个组中两个类的距离和这个组的大小（元素数量）计算分数
        scores = np.full((n_clusters, n_clusters), np.inf)
        for i, j in triu(n_clusters):
            scores[i, j] = distances[i, j] / avg_distance + self.relative_weight * np.sqrt((clusters[i].n + clusters[j].n) / avg_cluster_size)

        bar = ProgressBar(widgets=[Percentage(), Bar(), ETA()])
        bar.start(max_value=n_clusters-self.num_clusters)
        # 聚类目标没有达成
        while n_clusters != self.num_clusters:
            # 选出分数最低的组，(i,j)分别是这个组中两个类的编号
            i, j = min(triu(n_clusters), key=scores.__getitem__)
            # 将这两个类合并
            new_cluster = clusters[i].merge(clusters[j])
            # 删除原先的两个类
            del clusters[j], clusters[i]
            # 除了原先两个类以外其他类的编号
            new_idx = [idx for idx in range(n_clusters) if idx not in (i, j)]

            # 计算新类和其他类组合的分数
            s = []
            for c in clusters:
                s.append([distance(c.core, new_cluster.core) / avg_distance + self.relative_weight * np.sqrt((c.n + new_cluster.n) / avg_cluster_size)])
            s = np.array(s)

            # 将旧的分数矩阵和新类分数合并成新的分数矩阵
            scores = np.block([[scores[new_idx][:, new_idx], s], [np.full(len(s)+1, np.inf)]])
            # 将新类加入clusters
            clusters.append(new_cluster)
            n_clusters = len(clusters)

            # 更新进度条
            bar.update(len(points)-n_clusters)
        bar.finish()
        return [c.points for c in clusters]


if __name__ == '__main__':

    data = pd.read_excel("transformed.xlsx", index_col=0).iloc[:, 1].dropna().drop_duplicates()
    locations = []
    for item in data:
        lat, lon = map(float, item.split(","))
        locations.append(Location(lat, lon))
    cluster = EvenCluster(130, 0.5)
    results = cluster.fit(locations)
    with open("cluster_l1.txt", "w") as f:
        for c in results:
            f.write("\t".join(map(lambda x: "{},{}".format(x[0], x[1]), c)) + "\n")