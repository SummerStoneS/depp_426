from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from transform import get_location, split_location, fill_na_for_notnumber, convert_date, convert_time, format_time,\
    transform_orders, transform_cars, convert_plate_number
from map_visual2 import map_visual
from find_cbx import solve
from mosek.fusion import SolutionError
from xml2 import Problem, Vehicle, ResultReader, VehicleType, Service
from datetime import datetime
import os

# step 1
# 拿到历史数据，对历史数据做清理，查地址，对历史数据聚类, 生成停车点的经纬度


orders_columns = ['订单号', '最早接货时间', '最晚接货时间', '开单体积', '接货地址']
static_cars_columns = ["车牌号", "净空"]         # 上传所属分区，送货开始时间，送货结束时间
dynamic_cars_columns = ["车牌号", '订单号', '经度', '纬度', '净空', '取货状态']
historical_columns = ['运单号', '订单号', '最早接货时间', '最晚接货时间', '开单体积', '接货地址', '约车时间']
# 新的release的车要加在动态拼车里
addin_cars_columns = ["车牌号", "净空", "送货起始地址", "接货起始地址", "接货终止地址", "接货开始时间", "接货结束时间"]    # 新的release的车



def log(msg):
    with open("log.txt", "a") as f:
        f.write(msg)
        f.write("\n")

"""
    数据清理和检查
"""


def process_raw(runtime=1, historical_data_path='上海接货明细.xlsx', save_path='上海接货数据with经纬度.xlsx'):
    """
    :param runtime:
    :param historical_data_path:
    :return: 处理历史数据，
    """
    def clean_data(data):
        """
        :param data:
        :return: 对一天的数据做清理
        """
        data = data[data['运单号'] != '--'].loc[:, historical_columns].reset_index(drop=True)
        return data

    def read_data(start=1, end=27):
        month_data = pd.DataFrame()
        for i in range(start, end):
            sheet_name = '3-' + str(i)
            print("正在处理{}的数据".format(sheet_name))
            data = pd.read_excel(historical_data_path, sheetname=sheet_name)
            data = clean_data(data)
            data['日期'] = sheet_name
            month_data = pd.concat([month_data, data])
        month_data['日期'] = month_data['日期'].apply(lambda x: '2018-' + x)
        month_data['日期'] = pd.to_datetime(['日期'])
        month_data['星期几'] = month_data['日期'].apply(lambda x: x.weekday())
        return month_data

    def tag_location(month_data):
        # 高德api获取经纬度
        month_data = get_location(month_data, use_col='接货地址')
        month_data = month_data[month_data['经纬度'].notnull()]
        month_data = split_location(month_data)                  # 生成经纬度两列
        return month_data

    if runtime == 1:
        histor_data = read_data(start=1, end=27)        # 3.1-3.27
        histor_data = tag_location(histor_data)
        histor_data.to_excel(save_path)

    else:
        histor_data = pd.read_excel(save_path)
    return histor_data


# step 1
# 拿到历史数据，对历史数据聚类, 生成停车点的经纬度

class ClusterRunner:
    def __init__(self, cluster_num=26, minor_clusters=3):
        self.data = None
        self.cluster_num = cluster_num
        self.minor_clusters = minor_clusters
        self.center_list = None
        self.cluster_weight = None

    def run_cluster(self):
        """
        :param raw_data: 所有历史接货订单数据
        :param cluster_n: 26  第一次聚类类别数
        :return: 原始订单数据新增一列cluster
        """

        feature_matrix = np.array(self.data[['经度', '纬度']])
        clf = KMeans(n_clusters=self.cluster_num, max_iter=1000, random_state=43)
        clf.fit(feature_matrix)
        cluster_num = pd.Series(clf.labels_, name='cluster')
        print("每个类别的数量：")
        print(cluster_num.value_counts())
        self.data.reset_index(drop=True, inplace=True)
        self.data = pd.concat([self.data, cluster_num], axis=1)
        # with open('KMeans_model/clf.pickle', 'wb') as f:            # 保存模型
        #     pickle.dump(clf, f)
        center = clf.cluster_centers_
        center_list = []
        for i in center:
            center_list.append(i.tolist())
        return center_list

    def minor_cluster(self):
        """
        :param data: 所有历史接货订单数据+每个订单所属大类的类别cluster列
        :param cluster_n: 26 第一次聚类大类数，第二次聚类小类数
        :param minor_clusters: 3
        :return: 对每一个类再找几个质心，返回的是质心（停车点）的list
        """

        def region_cluster(raw_data):
            feature_matrix = np.array(raw_data[['经度', '纬度']])
            clf = KMeans(n_clusters=self.minor_clusters)
            clf.fit(feature_matrix)
            return clf.cluster_centers_

        center_list = []
        for i in range(self.cluster_num):
            cluster_data = self.data[self.data['cluster'] == i]
            center = region_cluster(cluster_data)
            for i in center:
                center_list.append(i.tolist())

        return center_list

    def fit_clusters(self, data, save_path='停车点_聚类中心location.xlsx'):
        self.data = data
        self.center_list = self.run_cluster()
        if self.minor_clusters > 0:
            self.center_list = self.minor_cluster()
        pd.Series(self.center_list).to_excel(save_path)
        return self.center_list

    def plot_map(self, data_day=None):
        """
        :param data_day: 如果没传日订单数据，则画历史订单聚类地图；如果传了日订单数据，画每日的订单分派停车点结果
        :return:
        """
        a, b = zip(*self.center_list)
        a, b = [str(x) for x in a], [str(x) for x in b]
        if not data_day:
            map_visual(self.data, cluster_n=self.cluster_num, centers=(a, b),
                   save_name='Pickup locations_{}大类{}小类一个月的数据.html'.format(self.cluster_num,self.minor_clusters))
        else:
            map_visual(data_day, cluster_n=self.cluster_num * self.minor_clusters, centers=(a, b),
                       save_name='Pickup locations_{}大类{}小类一天的数据.html'.format(self.cluster_num,self.minor_clusters))

    def load_centers(self, save_path='停车点-聚类中心location.xlsx'):
        self.center_list = pd.read_excel(save_path)
        self.center_list['经度'] = self.center_list[0].apply(lambda x: eval(x)[0])
        self.center_list['纬度'] = self.center_list[0].apply(lambda x: eval(x)[1])
        return self.center_list

    def predict(self, data_day, save_path=None):
        # 读取Model,预测某一天的数据
        # with open('KMeans_model/clf.pickle', 'rb') as f:
        #     clf = pickle.load(f)
        self.load_centers(save_path=save_file+filename_dict["depots_center_location"])

        def points_depots(daily_data, depots_center):
            """
            :param daily_data: 每一天的pickup订单
            :param depots_center: depots的经纬度,第一个元素是经度，第二个元素是纬度
            :return: 每一天订单新增一列cluster，属于哪个仓库
            """
            daily_cluster = daily_data.copy()
            try:
                del daily_cluster['cluster']
            except:
                pass
            for row, item in daily_data.iterrows():
                a = (depots_center - np.array(item[['经度', '纬度']])) ** 2
                distance = (a[:, 0]*3/4 + a[:, 1]) ** 0.5
                close_cluster = np.argmin(distance)
                daily_cluster.loc[row, 'cluster'] = close_cluster
            return daily_cluster

        data_day = points_depots(data_day, np.array(self.center_list[['经度', '纬度']]))
        if save_path:
            data_day.to_excel(save_path)
        return data_day

    @staticmethod
    def find_cluster_weights(data, date_col='date'):
        """
        :param data: 历史数据，一个月，一个季度，含有cluster列，which is每天每个订单所属的小区
        :param date_col: datetime类型的时间
        :return: 每个小区在14点后接货的概率中位数做weight
        """
        # data = pd.read_excel('上海接货数据with经纬度.xlsx')
        # data.loc[:, '约车时间'] = data['约车时间'].map(
        #     lambda x: datetime.datetime(1899, 12, 30, tzinfo=datetime.timezone.utc) + datetime.timedelta(days=x))
        data['hour'] = data[date_col].apply(lambda x: x.hour)
        data['date'] = data[date_col].apply(lambda x: x.date())
        data['book_ahead'] = 'No'
        data.loc[data[data['hour'] < 14].index, 'book_ahead'] = 'yes'
        grouped = data.groupby(['cluster', 'date', 'book_ahead'])['开单体积'].count()
        # 每个cluster 一周内 每天14点后订单的概率，没有的用0.01替代
        cluster_pop_prob = grouped.loc[(slice(None)), slice(None), 'No'] / data.groupby(['cluster', 'date'])[
            '开单体积'].count().fillna(0.01)  # grouped/每个cluster一天的总订单数
        cluster_weight = cluster_pop_prob.groupby(['cluster']).median()
        return cluster_weight

    def get_historical_weights(self, save_path='14点后订单概率的权重列表.xlsx'):
        """
        :return: 计算历史数据每天每笔订单所属哪个小区
        """
        self.data["约车时间"] = self.data["约车时间"].apply(fill_na_for_notnumber)
        self.data = self.data[self.data["约车时间"].notnull()]
        self.data['long_date'] = self.data['约车时间'].apply(lambda x: convert_date(x, 'date'))
        month_loadings = pd.DataFrame()
        for date in self.data["long_date"].dt.date.unique():
            daily_data = self.data[self.data['long_date'].dt.date == date]
            daily_cluster = self.predict(daily_data, save_path=None)
            month_loadings = pd.concat([month_loadings, daily_cluster])
        cluster_weight = self.find_cluster_weights(month_loadings, date_col='long_date')
        if save_path:
            cluster_weight.to_excel(save_path)

    def load_historical_weights(self, save_path='14点后订单概率的权重列表.xlsx'):
        self.cluster_weight = pd.read_excel(save_path, index_col=0)
        return self.cluster_weight
    
    def weekday_cluster_loads(self, data, weekday=1):
        """
        :param data: 某一段历史订单数据
        :param weekday: 星期几
        :return: Series, indexs是cluster，value是cluster loads
        """
        self.data = data
        weekday -= 1
        self.data.loc[:, '最早接货时间'] = self.data['最早接货时间'].apply(fill_na_for_notnumber)
        self.data = self.data[self.data["最早接货时间"].notnull()]
        self.data['date'] = self.data['最早接货时间'].apply(lambda x: convert_date(x, 'date'))
        self.data['day'] = self.data['最早接货时间'].apply(lambda x: convert_date(x, 'weekday'))
        self.data['开单体积'] = self.data['开单体积'].apply(fill_na_for_notnumber)
        self.data['开单体积'] = self.data['开单体积'].fillna(self.data['开单体积'].median())

        same_weekday_data = self.data[self.data["day"] == weekday]    # 找到历史同星期数据

        # 每一天都对所有订单看归属的cluster，计算cluster loads，最后计算平均值
        month_loads = pd.DataFrame()
        for date in same_weekday_data["date"].dt.date.unique():
            one_day_data = same_weekday_data[same_weekday_data["date"].dt.date == date]
            one_day_data = self.predict(one_day_data, save_path=None)
            oneday_cluster_loads = one_day_data.groupby("cluster")["开单体积"].sum()
            month_loads = pd.concat([month_loads, oneday_cluster_loads], axis=1)
        avg_cluster_loads = month_loads.mean(axis=1)                   # 历史上上同星期的日子每个cluster平均的loads
        return avg_cluster_loads

    """
        每天凌晨给每个cluster计算派哪些车
    """

    def find_cluster_cars(self, data, cars_info, weekday, save_path='每个区派车容积.xlsx'):
        """
        :param data: 历史数据
        :param cars_info:
        :param weekday:
        :param save_path:
        :return:
        """
        cars = np.array(cars_info['净空']).reshape(-1, 1)  # index 是取出来的sample原来的行号

        # 把今天没有订单的cluster补0
        cluster_loads = self.weekday_cluster_loads(data, weekday=weekday)    # 星期一历史每个小区的平均loads
        cluster_loads = pd.concat([self.cluster_weight, cluster_loads], axis=1)
        cluster_loads.iloc[:, 1].fillna(0, inplace=True)

        # 预判断这一天的车容积够不够用
        def pre_loads_satisfiy(loads, cars):
            if (len(cars) < len(loads)) or (cars.sum() < loads.sum()):  # 有load的区至少要有一辆车，总的load不能少于车的总capacity
                return False

            # 计算每个区的load是不是能被cover到
            loads = sorted(list(loads))  # 从小到大排序
            cars = sorted(list(cars))

            def cover_this_cluster(cluster_size, car_list):
                capacity_sum = car_list.pop()
                while capacity_sum < cluster_size:
                    if not len(car_list):  # 没有车能再补充capacity了
                        return True
                    capacity_sum += car_list.pop(0)  # 从最小的开始补充capacity
                return False

            while loads:
                if not cars or cover_this_cluster(loads.pop(), cars):  # load最大的那个能不能被cars中剩余的车辆满足
                    return False
            return True

        if not pre_loads_satisfiy(cluster_loads['开单体积'], cars):
            log("今天的车辆不能够满足每一个区的接货需求")
            return

        upper_bound = round((cars[:, 0].sum() - cluster_loads.iloc[:, 1].sum()) / self.cluster_num / self.minor_clusters)

        while 1:
            try:
                x = solve(cars, np.array(cluster_loads.iloc[:, 1]).reshape(-1, 1), np.array(self.cluster_weight).reshape(-1, 1),
                          constraint=upper_bound, timeout=180)  # cluster*车的矩阵
            except SolutionError:
                log("error with upper bound={}".format(upper_bound))
                upper_bound += 5
            else:
                log("completed with upper bound {}".format(upper_bound))
                break
        dispatch_cars = pd.DataFrame(x * cars.reshape(1, -1), index=cluster_loads.index, columns=cars_info['车牌号'])
        # print(np.dot(x, cars)-np.array(cluster_loads['sum']).reshape(-1, 1))      # print 剩余体积
        dispatch_cars.to_excel(save_path)
        # cluster_car_number = pd.DataFrame(x, index=cluster_loads.index, columns=cars_info['车牌号'])
        # round(cluster_car_number).to_excel(save_file+"大类{}小类{}".format(self.cluster_num, self.minor_clusters)+"每个区派车0-1矩阵.xlsx")
        return dispatch_cars

    def load_dispatched_cars(self, save_path=None):
        if save_path:
            try:
                dispatched_cars = pd.read_excel(save_file + "大类{}小类{}".format(self.cluster_num, self.minor_clusters)
                                                + save_path, index_col=0)
                return dispatched_cars
            except Exception as e:
                print(e)
        else:
            raise ValueError("please provide save_path")

    def load_daily_cluster(self, save_path='集中取货数据小区分派.xlsx'):
        return pd.read_excel(save_path)


def daily_route(daily_cluster, cluster_cars, depots_center, cluster_num=26, minor_clusters=3, save_path="今日静态路线规划.xlsx"):
    """
    :param daily_cluster: 订单网点分配方案.xlsx
    :param cluster_cars: 每个区派车容积，0-1矩阵用车容积替换1 index:cluster no. columns: 车牌号  content: 初始净空
    :param depots_center: 停车点经纬度 index:cluster no. columns: [[经度，纬度]，经度，纬度]
    :param cluster_num:
    :param minor_clusters:
    :return: 调用java的线路规划
    """
    routing_result = pd.DataFrame()         # 每一天所有cluster的routing拼在一起

    # 给每个cluster 规划线路
    for i in range(cluster_num*minor_clusters):
        log("正在规划cluster{}的线路".format(i))
        problem = Problem()
        if np.where(daily_cluster["cluster"].unique() == i)[0].size > 0:                      # 如果这个cluster有service，否则在数据中查不到
            cluster_i = daily_cluster[daily_cluster['cluster'] == i].reset_index(drop=True)   # 从接货数据选出每个小区
        else:
            continue

        # add a cluster's pickup services
        for j, values in cluster_i.iterrows():              # 给Jsprit添加小区里每一个要pickup的点
            # problem.add_service(Service('35', 'pickup', 'location1', (5.0, 35.0), 10, 90.0, [(0, "1.7976")]))
            # 添加取件信息
            start_time = values['最早接货时间']
            end_time = values['最晚接货时间']

            if convert_time(end_time) <= convert_time(start_time):
                log("时间信息错误")
                log(values["订单号"])
                continue                        # 最晚接货时间小于最早接货时间

            locationid = "[x={}][y={}]".format(values['经度'], values['纬度'])

            # 添加要去接的客户信息,默认取货的duration是1020 seconds
            problem.add_service(Service(values["订单号"], 'pickup', locationid, (values['经度'], values['纬度']),
                                        int(values['开单体积']*100), 1020.0, [(convert_time(start_time), convert_time(end_time))]))

        # add vehicle
        cluster_cars_i = cluster_cars.iloc[i, :]         # 第i个小区的派车方案
        selected_cars = cluster_cars_i[cluster_cars_i > 0]
        # 添加车辆信息
        for j, value in selected_cars.iteritems():                  # j:车牌号，value车容积
            problem.add_vehicle_type(VehicleType(j, int(value*100), {'fixed': 10000, 'distance': 2.0, 'time': 3.0}))
            problem.add_vehile(Vehicle(j, j, tuple(eval(depots_center.iloc[i, 0])),
                                       (121.263686, 31.195518), (18000, 79200)))        # id,车类型id，起始点，结束点，时间窗
        save_path2 = "./tmp/cluster_{}.xml".format(i)
        problem.to_xml(save_path2)
        result = os.popen("java -jar Jspirit-core-1.0-SNAPSHOT.jar " + save_path2)
        lines = []
        for line in result:
            print(line.strip("\n"))
            lines.append(line)
        dict_data = ResultReader.read(lines)
        print(dict_data)
        detailed_solution = dict_data['detailed solution']      # 拿规划数据
        detailed_solution['cluster'] = i
        detailed_solution.reset_index(drop=True, inplace=True)

        detailed_solution.loc[:, 'arrTime'] = detailed_solution['arrTime'].fillna(0).apply(format_time)   # 把距离零点的秒数改回时间
        detailed_solution.loc[:, 'endTime'] = detailed_solution['endTime'].fillna(0).apply(format_time)
        detailed_solution = pd.merge(detailed_solution, daily_cluster[["订单号", "接货地址"]], left_on='job', right_on='订单号', how='left')
        routing_result = pd.concat([routing_result, detailed_solution])
    routing_result.to_excel(save_path)


"""
    实时拼车：

    对早晨先送再接的车：

    大约每10分钟刷新一次系统，得到一拨新的待接订单信息。并且要从系统中调到车辆信息和加运力的信息
    （columns: 车牌号，待接订单列表，经纬度，剩余容积，是否已送完货的送货状态，是否愿意开始接货的接货状态），
    如果司机的状态是已经送完货（接货状态是开启的），则司机的定位是真实定位，时间窗是now()-20:00;
    如果司机的状态是没送完货且接货状态没开启，则司机的定位要改成他的接货小区的中心，时间窗是14:00-20：00；
    如果司机的状态是没送完货但是他开启了接货状态，则司机的定位是真实定位，时间窗是now()-20:00;
    把新的订单放到每个司机的订单列表中，比较哪个司机增加的成本最少，就拼入哪个司机的任务列表中，更新这个司机的路径规划；


    对门店11点半release出来的车：
    司机的真实定位，时间窗是now to 19:00，还要给司机造一个任务列表

"""


def add_a_service(problem, task):
    """
    :param problem: xml input handler
    :param task: 一条订单数据
    :return: 在xml中添加一段service
    """

    start_time = task['最早接货时间']
    end_time = task['最晚接货时间']

    if convert_time(end_time) <= convert_time(start_time):
        print("时间信息错误")
        print(task["订单号"])
        return None

    location_id = "[x={}][y={}]".format(task['经度'], task['纬度'])
    problem.add_service(Service(task["订单号"], 'pickup', location_id, (task['经度'], task['纬度']),
                                int(task['开单体积'] * 100), 1020.0, [(convert_time(start_time), convert_time(end_time))]))
    # 1020是17分钟，表示取货的duration


def add_old_services(jsprit_problem, driver_tasks_list, daily_cluster_data):
    """
    :param jsprit_problem:     调Jsprit用的类
    :param driver_tasks_list: 同一个司机的不同订单
    :param daily_cluster_data: 有订单信息的表
    :return: 把一个司机已经有的订单加到problem里
    """
    task_list = pd.merge(driver_tasks_list, daily_cluster_data, left_on='订单号', right_on='订单号', how='left')  # 按订单号补充订单信息
    for row, values in task_list.iterrows():
        add_a_service(jsprit_problem, values)

def get_cars_depots_center(dispatched_cars, depots_center):
    """
    :param dispatched_cars: index: cluster ，columns: 今天派的所有车的车牌号
    :param depots_center: index:cluster， columns: [0] 经纬度的list 是str格式
    :return: 每个车分配的停车点地址
    """
    dispatched_cars = dispatched_cars.T
    cars_depots = []
    cars_index = []
    for index, values in dispatched_cars.iterrows():
        center = depots_center.iloc[values[values != 0].index]
        cars_depots.append(tuple(eval(center.iloc[0, 0])))
        cars_index.append(index)
    return pd.Series(cars_depots, index=cars_index)


def dynamic_route(test, daily_cluster, cars_depots, drivers_data, save_path='动态派车结果.xlsx', with_addin=False):
    """
    :param test:            pop up的订单
    :param daily_cluster:   订单信息，用来查询司机剩余任务列表里的订单
    :param cars_depots:     每辆车对应的停车点
    :param drivers_data:    司机的实时信息 ["车牌号",	'订单号', '经度', '纬度', '净空', '送货状态', '取货状态']
    :return: 被更新线路的司机的新线路
    """

    final_route_replanning = {}             # 只返回线路会被更新的司机的新线路规划
    for i in range(len(test)):

        new_task = test.loc[i, :]       # 一条新订单
        log("正在为第{}个订单{}拼车".format(i, new_task['订单号']))
        new_task = pd.DataFrame(new_task).T

        new_task = get_location(new_task)       # 获得这个task的经纬度
        if new_task['经纬度'].isnull().any():
            log("订单{}无法匹配地址".format(new_task["订单号"]))
            continue
        new_task['经度'] = new_task['经纬度'].apply(lambda x: x.split(',')[0])
        new_task['纬度'] = new_task['经纬度'].apply(lambda x: x.split(',')[1])

        cost = {}                   # 记录每个司机添加订单后的成本
        route = {}                  # 记录每个司机添加订单后的线路规划

        for plate_num in drivers_data["车牌号"].unique():
            # 计算给每个司机拼入新业务后的成本

            # 添加这个司机还没完成的订单
            driver_data = drivers_data[drivers_data["车牌号"] == plate_num]
            location = (driver_data['经度'].iloc[0], driver_data['纬度'].iloc[0])
            capacity = driver_data["净空"].iloc[0]                # 到了下午车还剩余的净空
            status = driver_data["取货状态"].iloc[0]              # 1是开启取货，0是还没开启，如果送货=1则取货状态必须是1

            del driver_data['经度']
            del driver_data['纬度']

            driver_info = cars_depots[cars_depots.index == plate_num]
            if driver_info.size == 0:
                log("今日总派车数据中没有这个车牌号{}".format(plate_num))
                continue

            # 添加这个司机的车辆信息
            def add_driver(problem):
                now = datetime.now()
                now = now.hour * 60 * 60 + now.minute * 60 + now.second
                problem.add_vehicle_type(VehicleType(plate_num, int(capacity * 100), {'fixed': 10000, 'distance': 2.0, 'time': 3.0}))
                if (status == 1) or (now > 50400):         # 如果开启了取货状态或者现在已经到了14:00以后，TW是现在到晚上，经纬度是实时经纬度
                    problem.add_vehile(Vehicle(plate_num, plate_num, (float(location[0]), float(location[1])),
                                               (121.263686, 31.195518), (now, 72000)))  # id,车类型id，起始点，结束点，时间窗
                else:                   # 没送完货，且时间是14:00以前，经纬度是停车点的经纬度
                    future_location = cars_depots[cars_depots.index == plate_num].iloc[0]
                    problem.add_vehile(Vehicle(plate_num, plate_num, (float(future_location[0]), float(future_location[1])),
                                               (121.263686, 31.195518), (50400, 72000)))  # 14:00-20:00

            def run_jsprit(type='new'):
                problem = Problem()  # 新创建一个xml input模板
                if type == 'new':
                    add_a_service(problem, new_task.loc[0, :])  # 添加新增的订单的客户信息
                add_old_services(problem, driver_data, daily_cluster)
                add_driver(problem)

                save_path2 = "./tmp/{}{}.xml".format(type, plate_num)
                problem.to_xml(save_path2)
                result = os.popen("java -jar Jspirit-core-1.0-SNAPSHOT.jar " + save_path2)
                lines = []
                for line in result:
                    lines.append(line)
                dict_data = ResultReader.read(lines)
                # print(dict_data)
                return dict_data

            try:
                new_dict = run_jsprit(type='new')                           # 计算新增订单的线路规划
                route[plate_num] = new_dict['detailed solution']            # 每个司机都有一个新的线路方案，最后成本增加最少的司机线路会被更新
            except Exception:
                raise
                continue

            new_cost = new_dict["solution"]['value'][0]                 # 拿到线路规划的cost
            old_dict = run_jsprit(type='old')                           # 计算原来订单的cost
            old_cost = old_dict["solution"]['value'][0]
            cost[plate_num] = ((float(new_cost) - float(old_cost)), location, capacity, status) # 这个司机拼入新单后的成本提升

        # 如果需要考虑川流车release的车，需要给cost增加一些新的车牌号的数据
        def add_released_vehicles(released_cars_real, released_cars_info):
            """
            :param released_cars_real: 川流车实时信息，经纬度，剩余体积
            :param released_cars_info: 运力库，接货停止地址，接货开始时间，接货停止时间
            :return:
            """
            for i, row in released_cars_real.iterrows():
                plate_number = row["车牌号"]
                released_driver_data = drivers_data[drivers_data['车牌号'] == plate_number]
                r_location = (row['经度'], row['纬度'])                       # 起始点，实时经纬度
                r_capacity = row["净空"]                              # 实时的剩余体积，跟drivers_data的列名应该保持一致
                r_status = row["取货状态"]                            # 加进来的车默认取货状态都是1

                # 去运力库找到这个车的接货开始时间，接货停止时间
                vehicle_infos = released_cars_info[released_cars_info["车牌号"] == plate_number]
                start_time = convert_time(vehicle_infos['接货开始时间'].iloc[0])
                end_time = convert_time(vehicle_infos['接货停止时间'].iloc[0])
                end_place = vehicle_infos['经纬度']

                def add_released_driver(r_problem):
                    r_problem.add_vehicle_type(
                        VehicleType(plate_number, int(r_capacity * 100), {'fixed': 10000, 'distance': 2.0, 'time': 3.0}))
                    r_problem.add_vehile(Vehicle(plate_number, plate_number, (float(r_location[0]), float(r_location[1])),
                                                   (end_place[0], end_place[1]), (start_time, end_time)))  # id,车类型id，起始点，结束点，时间窗

                def released_jsprit(type='new'):
                    problem = Problem()  # 新创建一个xml input模板
                    if type == 'new':
                        add_a_service(problem, new_task.loc[0, :])                  # 添加新增的订单的客户信息
                    if len(released_driver_data):
                        add_old_services(problem, released_driver_data, daily_cluster)           # 添加旧订单
                    add_released_driver(problem)

                    save_path2 = "./tmp/{}{}.xml".format(type, plate_number)
                    problem.to_xml(save_path2)
                    result = os.popen("java -jar Jspirit-core-1.0-SNAPSHOT.jar " + save_path2)
                    lines = []
                    for line in result:
                        # print(line.strip("\n"))
                        lines.append(line)
                    dict_data = ResultReader.read(lines)
                    # print(dict_data)
                    return dict_data

                try:
                    r_new_dict = released_jsprit(type='new')  # 计算新增订单的线路规划
                    route[plate_num] = r_new_dict['detailed solution']  # 每个司机都有一个新的线路方案，最后成本增加最少的司机线路会被更新
                except Exception:
                    continue

                r_new_cost = r_new_dict["solution"]['value'][0]  # 拿到线路规划的cost
                if len(released_driver_data):
                    r_old_dict = released_jsprit(type='old')  # 计算原来订单的cost
                    r_old_cost = r_old_dict["solution"]['value'][0]
                else:
                    r_old_cost = 0
                cost[plate_number] = ((float(r_new_cost) - float(r_old_cost)), r_location, r_capacity, r_status)  # 这个司机拼入新单后的成本提升

        if with_addin:
            addin_vehicles_real = pd.read_excel(upload_path + filename_dict["addin_vehicles_real"])
            convert_plate_number(addin_vehicles_real, use_col='车牌号')
            addin_vehicles_info = pd.read_excel(upload_path + filename_dict["addin_vehicles_info"])
            convert_plate_number(addin_vehicles_info, use_col='车牌号')
            add_released_vehicles(addin_vehicles_real, addin_vehicles_info) # 如果上传了新的运力，加入新的运力

        if not cost:
            log("订单{}无法成功分派到现有车辆".format(new_task['订单号']))  # 如果cost是空字典，则所有的车都不能接此单
            continue
        # print(cost)
        best_cost = 500000
        best_plate = None
        best_location = None
        best_capacity = None
        best_status = None
        for key, value in cost.items():                            # 找到成本增加最少的车牌号,key是车牌号，value是增加的成本
            if value[0] < best_cost:
                best_cost = value[0]
                best_plate = key
                best_location = value[1]
                best_capacity = value[2]
                best_status = value[3]

        final_route_replanning[best_plate] = route[best_plate].reset_index(drop=True)  # 那个表现最好的车的路线被重新规划了

        # 给这个司机的未完成任务列表添加这个司机车牌号，订单号；给原有的订单pool增添这个订单的其他信息
        drivers_data.append(pd.DataFrame([[best_plate, new_task['订单号'], best_location[0], best_location[1],
                                           best_capacity, best_status]], columns=drivers_data.columns), ignore_index=True)
        daily_cluster.append(new_task[['订单号', '最早接货时间', '最晚接货时间', '开单体积', '接货地址', '经度', '纬度']], ignore_index=True)

    final_data = pd.DataFrame()
    for key, value in final_route_replanning.items():
        final_data = pd.concat([final_data, value])         # Jsprit输出结果自带车牌号
    # now = datetime.strftime(datetime.now(), '%Y-%m-%d %H:%M:%S')
    final_data.loc[:, 'arrTime'] = final_data['arrTime'].fillna(0).apply(format_time)  # 把距离零点的秒数改回时间
    final_data.loc[:, 'endTime'] = final_data['endTime'].fillna(0).apply(format_time)
    full_tasks = pd.concat([daily_cluster[['订单号', '接货地址']], test[['订单号', '接货地址']]])  # 补上中文地址名称
    final_data = pd.merge(final_data, full_tasks, how='left', left_on='job', right_on='订单号')
    final_data.to_excel(save_path)
    return final_data

"""
    main

"""


def initial_cluster(history_data, cluster_num=26, minor_clusters=3,
                    depots_center_path='停车点-聚类中心location.xlsx',
                    weights_path='14点后订单概率的权重列表.xlsx'):
    cluster = ClusterRunner(cluster_num=cluster_num, minor_clusters=minor_clusters)
    cluster.fit_clusters(history_data, save_path=depots_center_path)       # 得到停车点的坐标
    # cluster.plot_map()     # 画一个月的历史数据地图
    cluster.get_historical_weights(save_path=weights_path)           # mosek需要的每一个cluster的权重


def dispatch_cars_by_history(historical_orders, today_cars_capacity, weekday=2, cluster_num=26, minor_clusters=3, save_path='每个区派车容积.xlsx'):
    """
    :return:    每天凌晨，拿到新一天的派车数据，对历史同期(星期几)数据计算每个cluster的平均loads，计算当天每个
    cluster的派车方案，给每个车分配停车点后，就可以给送货程序确定每辆车的终点，即取货的停车点
    """
    # Step1:
    cars_info = today_cars_capacity
    cars_info = transform_cars(cars_info, type='static')

    cluster = ClusterRunner(cluster_num=cluster_num, minor_clusters=minor_clusters)
    cluster.load_centers(save_path=save_file+filename_dict["depots_center_location"])
    cluster.load_historical_weights(save_path=save_file+filename_dict["depots_weights"])
    cluster.find_cluster_cars(historical_orders, cars_info, weekday=weekday, save_path=save_path)       # 星期一


def static_vrp(static_pickup_data, runtime=2, cluster_num=26, minor_clusters=3, save_path="今日静态路线规划.xlsx"):
    # Step3:第一次规划取货路线(静态规划)，大约发生在中午12点。把12点前收集到的订单数据分派给现有车辆，
    # 车的起始点是车辆的接货小区中心，终止点是外场。于是每辆车都有一个待接任务列表。

    if runtime == 1:                                # 初次运行，调经纬度，调参的时候不要花时间。。。
        new_orders = static_pickup_data
        new_orders = transform_orders(new_orders)
        new_orders = get_location(new_orders)
        unfound_data = new_orders[new_orders['经纬度'].isnull()]
        if len(unfound_data) > 0:
            log('没有找到的静态订单已保存为[没有找到经纬度的.xlsx]')
            unfound_data.to_excel('没有找到经纬度的静态取货数据.xlsx')
        new_orders = new_orders[new_orders['经纬度'].notnull()]
        new_orders = split_location(new_orders)       # 给订单找经纬度
        new_orders.to_excel(upload_path + filename_dict["static_orders"])
        static_pickup_data = new_orders

    # 给每个订单分配小区
    cluster = ClusterRunner(cluster_num=cluster_num, minor_clusters=minor_clusters)
    depots_centers = cluster.load_centers(save_path=save_file+filename_dict["depots_center_location"])
    daily_cluster = cluster.predict(static_pickup_data, save_path=save_file + filename_dict["orders_with_clusters"])
    dispatch_cars = cluster.load_dispatched_cars(save_path=filename_dict["dispatch_cars_to_clusters"])
    daily_route(daily_cluster, dispatch_cars, depots_centers, cluster_num=cluster_num, minor_clusters=minor_clusters,
                save_path=save_path)


def dynamic_vrp(dynamic_pickup_data, realtime_drivers_state, runtime=2, cluster_num=26, minor_clusters=3,
                save_path='动态拼单线路规划.xlsx'):
    # 动态规划
    # 新加进来的一个batch的订单
    dynamic_orders = dynamic_pickup_data
    dynamic_orders = transform_orders(dynamic_orders)
    if runtime == 1:
        dynamic_orders = get_location(dynamic_orders)
        unfound_data = dynamic_orders[dynamic_orders['经纬度'].isnull()]
        if len(unfound_data) > 0:
            print('没有找到的已保存为[没有找到经纬度的.xlsx]')
            unfound_data.to_excel('动态取货没有找到经纬度的.xlsx')
        dynamic_orders = dynamic_orders[dynamic_orders['经纬度'].notnull()]
        dynamic_orders = split_location(dynamic_orders)       # 给订单找经纬度
        dynamic_orders.to_excel(filename_dict["dynamic_orders"])

    # 车的实时数据 ['车牌号', '剩余订单id', '经度',' 纬度'，'净空'，'送货状态', '取货状态']
    # 送货状态：真实情况把所有货都送完了，1是送完货；取货状态：司机自己开启了取货状态按钮，1是司机开启了取货按钮
    drivers_data = realtime_drivers_state
    drivers_data = transform_cars(drivers_data, type='dynamic')

    cluster = ClusterRunner(cluster_num=cluster_num, minor_clusters=minor_clusters)
    dispatch_cars = cluster.load_dispatched_cars(save_path=filename_dict["dispatch_cars_to_clusters"])
    daily_cluster = cluster.load_daily_cluster(save_path=save_file + filename_dict["orders_with_clusters"])
    depots_centers = cluster.load_centers(save_path=save_file+filename_dict["depots_center_location"])
    cars_target_position = get_cars_depots_center(dispatch_cars, depots_centers)
    dynamic_route(dynamic_orders, daily_cluster, cars_target_position, drivers_data, save_path, with_addin = is_addin)


if __name__ == '__main__':

    cluster_num, minor_clusters = 26, 3

    upload_path = './dist/'                                                         # 保存客户上传数据
    save_file = './dist/serena/大类{}小类{}/'.format(cluster_num, minor_clusters)   # 保存中间过程数据
    if not os.path.exists(save_file):
        os.makedirs(save_file)

    filename_dict = {
        "historical_data_name": "上海接货明细.xlsx",
        "processed_historical_data": '上海接货数据with经纬度.xlsx',
        "depots_center_location": '停车点-聚类中心location.xlsx',
        "depots_weights": '14点后订单概率的权重列表.xlsx',
        "dispatched_cars_capacity": "今日派车信息.xlsx",
        "dispatch_cars_to_clusters": '每个区派车容积.xlsx',
        "static_orders": "静态取货订单信息.xlsx",
        "orders_with_clusters": '集中取货数据小区分派.xlsx',
        "static_route_result": "今日静态路线规划.xlsx",
        "dynamic_orders": "tasks.xlsx",
        "real_time_cars_info": "cars.xlsx",
        "dynamic_route": "动态拼单线路规划.xlsx",
        "addin_vehicles_real": "川流车实时信息.xlsx",
        "addin_vehicles_info": "川流车运力.xlsx",

    }

    # Pre-Step: 根据历史数据生成停车点的经纬度，和优化派车方案时cluster的权重
    # historical_data = process_raw(runtime=2, historical_data_path=upload_path + filename_dict["historical_data_name"],
    #                               save_path=upload_path + filename_dict["processed_historical_data"])   # load historical orders data

    # initial_cluster(historical_data, cluster_num=cluster_num, minor_clusters=minor_clusters,
    #                 depots_center_path=save_file + filename_dict["depots_center_location"],  # 停车点的经纬度
    #                 weights_path=save_file + filename_dict["depots_weights"])       # 每个小区14:00后订单的概率，概率越大的，
    #                                                                                 # 在派车的时候要多派点容积

    # Step2: 凌晨生成派车方案
    # today_car_capacity = pd.read_excel(upload_path + filename_dict["dispatched_cars_capacity"])     # 今天的运力
    # day_of_week = 2     # 星期二
    # dispatch_cars_by_history(historical_data, today_car_capacity, weekday=day_of_week, cluster_num=cluster_num,
    #                          minor_clusters=minor_clusters,
    #                          save_path=save_file + "大类{}小类{}".format(cluster_num, minor_clusters) + filename_dict["dispatch_cars_to_clusters"])

    # # Step3: 第一次规划取货路线(静态规划) 经纬度重新计算的话需要runtime=1
    # initial_pickups = pd.read_excel(upload_path + filename_dict["static_orders"])
    # static_vrp(initial_pickups, runtime=2, cluster_num=cluster_num, minor_clusters=minor_clusters,
    #            save_path=filename_dict["static_route_result"])

    # Step4: 动态拼单
    dynamic_pickups = pd.read_excel(upload_path + filename_dict["dynamic_orders"])  # 拼单

    is_addin = True                             # 如果上传了川流车的运力

    realtime_drivers_state = pd.read_excel(upload_path + filename_dict["real_time_cars_info"])  # 早上送货的车的实时状态
    dynamic_vrp(dynamic_pickups, realtime_drivers_state, runtime=2, cluster_num=cluster_num, minor_clusters=minor_clusters,
                save_path=filename_dict["dynamic_route"])  # runtime=1是要不要重新计算经纬度



