import pandas as pd
from VRP_ALL import ClusterRunner, filename_dict, dispatch_cars_by_history
from transform import convert_date, fill_na_for_notnumber, get_location, split_location
import numpy as np
from itertools import chain
from collections import defaultdict
import json
from datetime import datetime

# address = pd.DataFrame()
# for i in range(1, 9):
#     filename = "delivery{}.csv".format(i)
#     f = open(filename, encoding='GB18030')
#     res = pd.read_csv(f)
#     res.columns = ["单号", "送货地址"]
#     address = pd.concat([address, res])


def count_weekday_numbers(history):
    """
    :param history: 历史数据
    :return: 历史数据中有几个星期0,1,2,3....6
    """
    history['weekday'] = history['约车时间'].apply(lambda x: convert_date(x, type='weekday'))
    history['order_date'] = history['约车时间'].apply(lambda x: convert_date(x, type=None).date())
    weekday_num = history.groupby(['weekday'])['order_date'].apply(lambda x: len(x.unique()))
    return weekday_num, history


def weekday_vehicles_simulate(weekday, history):
    """
    :param weekday: 星期几
    :param history: 历史数据，带车牌号和净空
    :return: 每个星期几每种车型平均数量，再生成总的运力
    """
    weekday_number, history = count_weekday_numbers(history)
    history_weekday = history[history["weekday"] == weekday]
    unique_plates = history_weekday.drop_duplicates(subset=['order_date', '车牌号'])
    unique_plates['净空'] = unique_plates['净空'].apply(fill_na_for_notnumber).fillna(14)
    cartypes_demand = (unique_plates["净空"].value_counts()/weekday_number.loc[weekday]).astype(int)
    cartypes_demand = cartypes_demand[cartypes_demand > 0]
    total_capacity = [[capacity]*number for capacity, number in cartypes_demand.iteritems()]
    total_capacity = pd.DataFrame(list(chain(*total_capacity))).reset_index()
    total_capacity.columns = ["车牌号", "净空"]
    return total_capacity


def depots_vehicles_capacity(runtime=1):
    """
    :return: 生成每个停车点平均带的14容积车的数量
    """
    if runtime == 1:
        # 计算每天，每个点上的车辆数和capacity，最后求平均
        historical_data = pd.read_excel(upload_path + filename_dict["processed_historical_data"])
        cluster_dict = defaultdict(list)
        for week_day in range(0, 7):
            daily_cars = weekday_vehicles_simulate(week_day, historical_data)
            depots_cars = dispatch_cars_by_history(historical_data, daily_cars, weekday=week_day, cluster_num=cluster_num, minor_clusters=minor_clusters,
                                         save_path=None)
            # depots_cars.to_excel('_'.join([str(week_day), "depots_cars.xlsx"]))
            # depots_cars = pd.read_excel('_'.join([str(week_day), "depots_cars.xlsx"]), index_col=0)
            cluster_carnum = (depots_cars.sum(axis=1)/14).astype(int)

            for cluster, car_num in cluster_carnum.iteritems():
                cluster_dict[str(cluster)].append(car_num)

        for cluster, carnum_list in cluster_dict.items():
            cluster_dict[cluster] = int(round(np.mean(carnum_list)))

        file = open('cluster_carnum.json', 'w')
        json.dump(cluster_dict, file)
        file.close()
    else:
        cluster_json = json.load(open('cluster_carnum.json'))
        cluster_dict = dict((int(key), value) for key, value in cluster_json.items())
    return cluster_dict


class Cluster:
    """
    一个类，包含地点、中心
    """

    def __init__(self, points):
        self.points = points                        # 每个类里所有点
        self.core = tuple(np.median(points, 0))     # 中心是所有点的坐标的中位数,tuple
        self.n = len(self.points)                   # 每个类里的点数
        self._capacity = None                       # 每个类包含的14容积车辆数
        self._orders = None                         # 每个类包含的货物体积综合
        self._depots_list = None                    # 每个类包含的停车点的tuple(坐标)list

    @property
    def capacity(self):
        return self._capacity

    @capacity.setter
    def capacity(self, val):
        self._capacity = val

    @property
    def orders(self):
        return self._orders

    @orders.setter
    def orders(self, val):
        self._orders = val

    @property
    def depots_list(self):
        return self._depots_list

    @depots_list.setter
    def depots_list(self, val):
        self._depots_list = val

    def merge(self, other):
        """
        将两个类合并成第三个类
        """
        return Cluster(self.points + other.points)


def depots_cluster(depots_center_list, cluster_list, type='depots'):
    """
    :param depots_center_list: list of tuple
    :param cluster_list: list of cluster object
    :param type:
    :return: 每个停车点属于哪个cluster，每个cluster包含的depots的列表：dict
    """
    # 计算所有cluster的停车点
    depots_cluster_distance_list = []
    for depots_center in depots_center_list:
        cluster_distance_list = []
        for cluster_i in cluster_list:
            cluster_distance_list.append(distance(depots_center, cluster_i.core))
        depots_cluster_distance_list.append(cluster_distance_list)
    depots_cluster_distance_matrix = np.array(depots_cluster_distance_list)
    depots_cluster_idx = np.argmin(depots_cluster_distance_matrix,axis=1)
    cluster_depots_dict = defaultdict(list)
    for depots_idx, cluster in enumerate(depots_cluster_idx):
        cluster_depots_dict[cluster].append(depots_idx)
    pd.Series(depots_cluster_idx).to_excel(''.join(["每个", type, "属于哪个cluster.xlsx"]))
    return cluster_depots_dict


def get_capacity(depots_capacity_dict, depots_center_list, cluster_list):
    """
    :param depots_capacity_dict:
    :param depots_center_list:
    :param cluster_list:
    :return: 给每个cluster找到capacity
    """
    cluster_depots_dict = depots_cluster(depots_center_list, cluster_list)
    # 计算每个cluster包含的14容积车的数量*14
    for cluster_i in range(len(cluster_list)):
        depots_list = cluster_depots_dict.get(cluster_i, [])
        depots_capacity = 0.00001
        depot_centers = []
        for depots_idx in depots_list:
            depots_capacity += depots_capacity_dict[depots_idx]
            depot_centers.append(depots_center_list[depots_idx])
        cluster_list[cluster_i].capacity = depots_capacity*14
        cluster_list[cluster_i].depots_list = depot_centers
        # assert all(x.capacity for x in cluster_list)
    return cluster_list


def get_delivery_loads(delivery_center_list, order_weight_list, cluster_list):
    """
    :param delivery_center_list: list of typle
    :param order_weight_list: list of order_weight/float
    :param cluster_list: list of cluster object
    :return: 每个cluster的货量
    """
    cluster_orders_dict = depots_cluster(delivery_center_list, cluster_list, type='delivery_orders')

    for cluster_i, orders_list in cluster_orders_dict.items():
        orders_weight = 0
        for orders_idx in orders_list:
            orders_weight += order_weight_list[orders_idx]
        cluster_list[cluster_i].orders = orders_weight
    return cluster_list


def triu(n):
    """
    生成n阶方阵的上三角的下标
    """
    for i in range(n):
        for j in range(i + 1, n):
            yield i, j


def distance(center1, center2):
    """
    :param center1: 某个类的质心，tuple
    :param center2: 某另一个类的质心，tuple
    :return: 两个质心的距离
    """
    a = (np.array(center1) - np.array(center2)) ** 2
    return (a[0] * 3 / 4 + a[1]) ** 0.5


class EvenCluster:
    """
    聚类，使得每个类的装载率尽量均等
    """

    def __init__(self, num_clusters, size_weight=0.4, cover_weight=0.9):
        """
        Parameters
        ==========
        num_clusters: int
            聚类的数量
        cover_weight: float
            类的capacity要能cover货量相对类间距离的权重
        load_weight: float
            类的车均票数权重
        """
        self.num_clusters = num_clusters
        self.cover_weight = cover_weight
        self.size_weight = size_weight

    def fit(self, points):
        """
        开始聚类

        Parameters
        ==========
        points: List[Location]
            点的列表，
        """
        clusters = [Cluster([point]) for point in points]  # 一开始每个点都是一个独立的类
        n_clusters = len(clusters)
        distances = np.zeros((n_clusters, n_clusters))  # 计算类之间两两的距离

        # 计算类之间的平均距离
        avg_distance = 0
        for i, j in triu(n_clusters):
            distances[i, j] = distance(clusters[i].core, clusters[j].core)
            avg_distance += distances[i, j]
        avg_distance /= n_clusters * (n_clusters - 1) / 2

        # 类的平均
        avg_cluster_size = len(points) / self.num_clusters

        # 两两类分组，根据每个组中两个类的距离和这个组的大小（元素数量）计算分数
        scores = np.full((n_clusters, n_clusters), 10000)
        new_scores = np.full((n_clusters, n_clusters), 10000)
        for i, j in triu(n_clusters):
            scores[i, j] = distances[i, j] / avg_distance + self.size_weight * np.sqrt(
                (clusters[i].n + clusters[j].n) / avg_cluster_size)
        #
        # bar = ProgressBar(widgets=[Percentage(), Bar(), ETA()])
        # bar.start(max_value=n_clusters - self.num_clusters)
        # avg_loads_coverage = (sum(orders_loads_list)-sum([car_num*14 for car_num in depots_carnum_dict.values()])) / self.num_clusters

        # TODO 改成 用上海总的送货量/num_clusters
        delivery_data = get_delivery_data(runtime=2)
        more_regularization_bar = round(len(delivery_data) / (cluster_num*1.2))

        print("加入正则项的bar：{}".format(more_regularization_bar))

        # 聚类目标没有达成
        while n_clusters != self.num_clusters:
            print("类数{}".format(n_clusters))
            # 选出分数最低的组，(i,j)分别是这个组中两个类的编号
            i, j = min(triu(n_clusters), key=new_scores.__getitem__)
            # 将这两个类合并
            new_cluster = clusters[i].merge(clusters[j])
            # 删除原先的两个类
            del clusters[j], clusters[i]
            # 除了原先两个类以外其他类的编号

            new_idx = [idx for idx in range(n_clusters) if idx not in (i, j)]

            # 计算新类和其他类组合的分数
            s = []
            for c in clusters:
                s.append([distance(c.core, new_cluster.core) / avg_distance + self.size_weight * np.sqrt(
                    (c.n + new_cluster.n) / avg_cluster_size)])
            s = np.array(s)

            # 将旧的分数矩阵和新类分数合并成新的分数矩阵
            scores = np.block([[scores[new_idx][:, new_idx], s], [np.full(len(s) + 1, np.inf)]])
            # 将新类加入clusters
            clusters.append(new_cluster)
            n_clusters = len(clusters)

            if n_clusters <= more_regularization_bar:
                # 计算每个类包含的货量，计算每个类包含的停车点，找到每个停车点包含的车辆数和capacity和
                get_capacity(depots_carnum_dict, depots_center_list, clusters)
                get_delivery_loads(orders_center_list, orders_loads_list, clusters)

                for i, j in triu(n_clusters):
                    bad_ratio = max(clusters[i].orders/clusters[i].capacity, clusters[j].orders/clusters[j].capacity)
                    new_ratio = (clusters[i].orders + clusters[j].orders)/(clusters[i].capacity + clusters[j].capacity)
                    new_scores[i, j] = scores[i, j] + self.cover_weight * new_ratio/bad_ratio
            else:
                new_scores = scores

        #     # 更新进度条
        #     bar.update(len(points) - n_clusters)
        # bar.finish()
        # 计算最后聚类效果
        for cluster_i, c in enumerate(clusters):
            print("类编号:{}\t货量:{}\t车容积：{}\t货比车：{}\t包含点数：{}".format(cluster_i, c.orders, c.capacity,
                                                                  c.orders/c.capacity, c.n))
        return [c.points for c in clusters], [c.depots_list for c in clusters], [c.core for c in clusters]


def get_center_list(depots_center_df):
    """
    :param depots_center_df: dataframe 有经度纬度列
    :return:
    """
    depots_list = []
    for row, values in depots_center_df.iterrows():
        depots_list.append((values['经度'], values['纬度']))
    return depots_list


def get_delivery_data(runtime=1, time_col='派送装车开始时间', start_dt=datetime(2018, 3, 1), end_dt=datetime(2018, 3, 26)):
    if runtime == 1:
        data = pd.read_excel('./dist/3月4月送货数据加地址.xlsx')
        data['long_date'] = pd.to_datetime(data[time_col])
        use_data = data[(start_dt <= data['long_date']) & (data['long_date'] <= end_dt)]
        data_locations = get_location(use_data, use_col='送货地址')
        data_locations = data_locations[data_locations['经纬度'].notnull()]
        data_locations = split_location(data_locations)                  # 生成经纬度两列
        data_locations.to_excel('送货数据with经纬度.xlsx')
    else:
        data_locations = pd.read_excel('送货数据with经纬度.xlsx')
    return data_locations[:2000]


def get_orders_location_loads(runtime=2, weight_col='体积'):
    """
    :param runtime: 1：原始数据处理，查询高德经纬度；2：直接读入处理好的数据
    :param weight_col: 货的体积的列名
    :return:
    """
    delivery_data = get_delivery_data(runtime=runtime, time_col='派送装车开始时间')
    orders_centers_list = get_center_list(delivery_data)
    loads_list = delivery_data[weight_col].tolist()
    return orders_centers_list, loads_list


def save_to_dict(points_list, depots_list, core_list):
    result_dict = defaultdict(dict)
    for delivery_cluster_i in range(len(points_list)):
        points_location = points_list[delivery_cluster_i]
        depots_location = depots_list[delivery_cluster_i]
        core_location = core_list[delivery_cluster_i]
        result_dict[str(delivery_cluster_i)] = dict(points=points_location, depots=depots_location, core=core_location)
    file = open('delivery_cluster_result.json', 'w')
    json.dump(result_dict, file)
    file.close()
    return result_dict


def result_dict_to_tw(result_dict, depots_center_df):
    tw_result_list = []
    for cluster_i in range(len(result_dict)):
        core_location = result_dict[cluster_i]['core']
        core_lon = core_location[0]
        core_lat = core_location[1]
        for depots_location in result_dict[cluster_i]['depots']:
            depots_id = depots_center_df[(depots_center_df["经度"] == depots_location[0])
                                         & (depots_center_df["纬度"] == depots_location[1])].index[0]
            tw_result_list.append({
                "depotId": depots_id,
                "longitude": depots_location[0],
                "latitude": depots_location[1],
                "district": dict(districtId=cluster_i, longitude=core_lon, latitude=core_lat)
            })
    return tw_result_list


if __name__ == '__main__':
    cluster_num, minor_clusters = 26, 3     # 接货聚停车点参数
    start_dt = datetime(2018, 3, 1)         # 送货数据开始日期
    end_dt = datetime(2018, 3, 26)          # 送货数据结束日期

    upload_path = './dist/'  # 保存客户上传数据
    save_file = './dist/serena/大类{}小类{}/'.format(cluster_num, minor_clusters)

    depots_carnum_dict = depots_vehicles_capacity(runtime=2)
    cluster = ClusterRunner(cluster_num=cluster_num, minor_clusters=minor_clusters)
    depots_centers = cluster.load_centers(save_path=save_file + filename_dict["depots_center_location"])
    depots_center_list = get_center_list(depots_centers)

    orders_center_list, orders_loads_list = get_orders_location_loads(runtime=2)

    # # depots_center_list = [(1, 2), (-1, -2), (2, 5), (-2, 4), (-3, -5), (3, -8)]
    # # orders_center_list = [(2, 6), (2, 7), (1, 9), (-1, 5), (-3, 2), (-3, -4), (-1, -5), (3, -4), (2, -1)]
    # # orders_loads_list = [2, 3, 1, 0.4, 3, 2, 4, 1, 3]
    #
    cluster = EvenCluster(9, size_weight=0.05, cover_weight=1.0)
    cluster_points_list, cluster_depots_list, cluster_core_list = cluster.fit(orders_center_list)
    cluster_result_dict = save_to_dict(cluster_points_list, cluster_depots_list, cluster_core_list)

    tw_result_dict_list = result_dict_to_tw(cluster_result_dict, depots_centers)






