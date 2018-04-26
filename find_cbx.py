import time
from mosek.fusion import *
import pandas as pd
import numpy as np
import threading


# def solve(cars, clusters, density, timeout=0):
def solve(cars, clusters, density, constraint=40, timeout=0):
    M = Model("binary_programming")                                 # 创建模型
    M.setSolverParam("mioDisableTermTime", 30)                      # 一堆参数，放宽约束条件，提前终止迭代
    M.setSolverParam("mioTolAbsRelaxInt", 0.01)
    M.setSolverParam("mioNearTolAbsGap", 0.1)
    M.setSolverParam("mioNearTolRelGap", 0.1)
    x = M.variable(Domain.binary(len(clusters), len(cars)))         # 0-1变量矩阵，行是区数，列是车数

    capacity = Expr.mul(x, cars)                     # 每个区分配到的车的capacity之和
    excess_capacity = Expr.sub(capacity, clusters)   # capacity - load
    # density越大的地方更有概率会出临时单，希望车更空一点
    # print(density.shape, clusters.shape)
    objective = Expr.dot(density, excess_capacity)       # sum(density * (capacity - load))
    # objective = Expr.dot(excess_capacity, density) - 0.1 * Expr.dot(excess_capacity, excess_capacity)
    M.constraint(Expr.sum(excess_capacity, 1), Domain.lessThan(constraint))
    M.constraint(excess_capacity, Domain.greaterThan(0))            # 约束1
    M.constraint(Expr.sum(x, 0), Domain.equalsTo(1))                # 约束2       # 每辆车都得用上，且用一次，不能跨区
    M.objective(ObjectiveSense.Maximize, objective)                 # 最大化目标函数
    T = threading.Thread(target=M.solve)
    T0 = time.time()
    try:
        T.start()
        while 1:
            if not T.is_alive():
                # raise RuntimeError("Solver teminated before anyting happened!")
                print("OK")
                break
            elif time.time() - T0 > timeout > 0:
                print("Timeout")
                M.breakSolver()
                break
    except KeyboardInterrupt:
        M.breakSolver()
    finally:
        try:
            T.join()
        except:
            pass
    # M.solve()                                                       # 求解问题
    return np.array(x.level()).reshape(len(clusters), len(cars))    # 返回解


def find_weight(data):
    """
    :param data:
    :return: 每个小区在14点后接货的weight
    """
    # data = pd.read_excel('上海接货数据with经纬度.xlsx')
    # data.loc[:, '约车时间'] = data['约车时间'].map(
    #     lambda x: datetime.datetime(1899, 12, 30, tzinfo=datetime.timezone.utc) + datetime.timedelta(days=x))
    data['hour'] = data['约车时间'].apply(lambda x: x.hour)
    data['date'] = data['约车时间'].apply(lambda x: x.date())
    data['book_ahead'] = 'No'
    data.loc[data[data['hour'] < 14].index, 'book_ahead'] = 'yes'
    grouped = data.groupby(['cluster', 'date', 'book_ahead'])['开单体积'].count()
    grouped = grouped[grouped['book_ahead'] == 'No']    # 每个cluster 14:00以后订单的数量
    # 每个cluster 一周内 每天14点后订单的概率，没有的用0.01替代
    cluster_pop_prob = grouped.loc[(slice(None)), slice(None), 'No']/data.groupby(['cluster', 'date'])['开单体积'].count().fillna(0.01)   # grouped/每个cluster一天的总订单数
    cluster_weight = cluster_pop_prob.groupby(['cluster']).median()
    return cluster_weight


if __name__ == '__main__':
    today = '2018-03-12'
    cluster_num = 26
    minor_clusters = 3
    cars_info = pd.read_excel("派车方案\{}\大类{}小类{}车辆信息.xlsx".format(today, cluster_num, minor_clusters))
    # cars_info = pd.read_excel("C:\\Users\\Ruofei Shen\\Desktop\\减少的车辆.xlsx")
    cars = np.array(cars_info['净空']).reshape(-1, 1)

    cluster_weight = pd.read_excel("派车方案\大分区{}小网点{}14点后订单概率的权重列表.xlsx".format(cluster_num, minor_clusters), index_col=0)
    density = cluster_weight

    cluster_loads = pd.read_excel("派车方案\{}\大类{}小类{}每个小区的loads.xlsx".format(today, cluster_num, minor_clusters))

    print(cars.shape, density.shape, cluster_loads['sum'].shape)
    x = solve(cars, np.array(cluster_loads['sum']).reshape(-1,1), np.array(density).reshape(-1,1))
    print(np.dot(x, cars)-np.array(cluster_loads['sum']).reshape(-1, 1))
    print(density)

    # data = pd.read_excel('今日订单网点120分配方案.xlsx')
    #
    #
    # def fill_na_for_notnumber(col):
    #     try:
    #         a = float(col)
    #     except:
    #         a = np.nan
    #     return a
    # data['开单体积'] = data['开单体积'].apply(fill_na_for_notnumber)
    # data['adjust_volume'] = data['开单体积'].fillna(data['开单体积'].median())
    #
    # # 计算每个cluster的订单数量和体积之和
    # cluster = data.groupby(['cluster'])['adjust_volume'].agg(['count', 'sum'])
    #
    # cars = np.array([np.random.choice([10, 20, 30, 40], size=1, p=[0.4, 0.3, 0.2, 0.1]) for i in range(80)])
    # cluster_loads = np.array(cluster['sum'])          # 每个类需要的开单体积
    # density = np.array(cluster['count']).astype(float)                   # 每个类里开单体积的数量
    #
    # x = solve(cars, cluster_loads, density)
    # print(np.dot(x, cars)-cluster_loads.reshape(-1, 1))
    # print(density)