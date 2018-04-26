import pandas as pd
from time import sleep
from amap import AMap
from tqdm import tqdm
import numpy as np
import datetime
import re

"""
    get_location：   将所有取货地址按照中文地址名字转化成经纬度，查不到的返回np.nan

    split_location： 把上一步经纬度列解析成经度和纬度两列

    转换时间

"""
orders_columns = ['订单号', '最早接货时间', '最晚接货时间', '开单体积', '接货地址']
static_cars_columns = ["车牌号", "净空"]
dynamic_cars_columns = ["车牌号",	'订单号', '经度', '纬度', '净空', '取货状态']


def get_location(source_data, use_col='接货地址'):
    """
    :param source_data:
    :param use_col:
    :param save_file:
    :return: 根据接货地址，调用高德api，返回一列新增的经纬度列
    """
    data = source_data[use_col].drop_duplicates()      # 避免查询重复的地址
    # api = AMap('d928d8749358e9245c9cec5c06aa9d06')        # 高德地图的key
    api = AMap('b9cacaf85e70270ab2f15e81b2a942ce')     # xulei的API

    result = []
    for item in tqdm(data):
        try:
            resp = api.geocode(item, '上海')
            result.append(resp['location'])
        except:
            result.append(np.nan)                   # 没有找到地址的返回nan
            # print(item)
        sleep(0.05)

    location = pd.Series(result, name='经纬度')
    data = data.reset_index(drop=True)
    data = pd.concat([data, location], axis=1)
    if "经纬度" in source_data.columns:
        del source_data['经纬度']
    location_data = pd.merge(source_data, data, left_on="接货地址", right_on="接货地址", how='left')
    return location_data


def split_location(raw_data, use_col='经纬度'):
    """
    :param raw_data:
    :param use_col:
    :return: 原数据增加经度一列，纬度一列
    """
    raw_data['纬度'] = raw_data[use_col].apply(lambda x: x.split(',')[1]).astype(float)
    raw_data['经度'] = raw_data[use_col].apply(lambda x: x.split(',')[0]).astype(float)
    return raw_data


def log(msg):
    with open("log.txt", "a") as f:
        f.write(msg)
        f.write("\n")


def convert_time(x):
    """
    :param x: 时间
    :return: 计算距离当日0点的秒数，满足Jsprit对输入时间的要求
    """
    b = datetime.datetime(1899, 12, 30, tzinfo=datetime.timezone.utc) + datetime.timedelta(
        days=x)
    time = b.time()
    return time.hour * 60 * 60 + time.minute * 60 + time.second  # 距离当天零点对应的秒数


def convert_date(x, type='weekday'):
    b = datetime.datetime(1899, 12, 30, tzinfo=datetime.timezone.utc) + datetime.timedelta(days=x)
    if type == 'weekday':
        return b.weekday()
    else:
        return b


def format_time(x):
    """
    :param x: Jsprit 跑出来的5位数时间
    :return: 转换成%H:%M:%S
    """
    x = float(x)
    if x != 0:
        hours = x / 60 / 60
        hour = int(hours)
        minutes = (hours - int(hours)) * 60
        minute = int(minutes)
        seconds = (minutes - minute) * 60
        second = int(seconds)
        return "{}:{}:{}".format(hour, minute, second)
    else:
        return np.nan


def fill_na_for_notnumber(col):
    try:
        a = float(col)
    except:
        a = np.nan
    return a


def check_columns(data, type='orders'):
    """
    :param data: 待接单信息或者是车辆信息
    :param type:
    :return: 检查该有的列名是不是都有
    """
    if type == 'orders':
        included_cols = orders_columns
    elif type == 'static_cars':
        included_cols = static_cars_columns
    elif type == 'dynamic_cars':
        included_cols = dynamic_cars_columns
    flag = None
    for column_name in included_cols:
        if column_name not in data.columns:
            log("{}没有{}列，请检查列名".format(type, column_name))
            flag = 1
    if flag:
        raise ValueError
    else:
        return data[included_cols]


def transform_orders(orders):
    orders = check_columns(orders)

    orders['开单体积'] = orders['开单体积'].apply(fill_na_for_notnumber)  # 处理没有体积信息的订单
    orders['开单体积'] = orders['开单体积'].fillna(0.4)
    return orders


def convert_plate_number(data, use_col='车牌号'):
    """
    :param data:
    :param use_col: 车牌号
    :return: 去掉车牌号前的“沪”字
    """
    data[use_col] = data[use_col].apply(lambda x: re.sub(u'[\u4E00-\u9FA5]+', '', x))


def transform_cars(cars, type='static'):
    if type == 'static':
        cars = check_columns(cars, type='static_cars')                                             # 检查列名
        cars_info = cars[['车牌号', '净空']].drop_duplicates().reset_index(drop=True)  # 车牌号不能重复
        if len(cars_info) != len(cars):
            log("有重复的车牌号，请修改后再次上传")
            raise ValueError
    elif type == 'dynamic':
        cars = check_columns(cars, type='dynamic_cars')
        cars_info = cars

    cars_info['净空'] = cars_info['净空'].apply(fill_na_for_notnumber)                  # 处理没有净空的车
    cars_info['净空'].fillna(14.51, inplace=True)
    convert_plate_number(cars_info, use_col='车牌号')                                  # 去掉车牌号前的“沪”字
    return cars_info


if __name__ == '__main__':
    pass