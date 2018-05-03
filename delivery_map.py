"""
@time: 5/2/2018 3:42 PM

@author: 柚子
"""

import plotly
from plotly.graph_objs import *
import random
import json


# mapbox_access_token = 'pk.eyJ1Ijoic3VtbWVyc3RvbmVzIiwiYSI6ImNqZmFrMmZzdDBhcXUyem1zdWl2N20ycWsifQ.46i6Xy2HDRhBArwqWGUd6Q'
mapbox_access_token = 'pk.eyJ1Ijoic25vd3dhbGtlcmoiLCJhIjoiY2piNGgxOWwyMnNvbzMyczc5aXgwbTY5dCJ9.swOGp-XV74Eh0Ow3ErtEXQ'


def map_visual(cluster_dict, cluster_n=9, delivery_points=True, save_name='Shanghai pickup locations'):
    cluster_data = []
    for i in range(cluster_n):
        lon, lat = zip(*cluster_dict[i]['points'])
        site_lon = lon
        site_lat = lat
        color = 'rgb'+str((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
        if delivery_points:
            cluster_data.append(
                Scattermapbox(
                    lat=site_lat,
                    lon=site_lon,
                    mode='markers',
                    marker=Marker(
                        size=5,
                        color=color,
                    ),
                    # text=locations_name,
                    hoverinfo='none'
                )
            )
        try:
            depots_lon, depots_lat = zip(*cluster_dict[i]['depots'])
            depots = Scattermapbox(
                lat=depots_lat,
                lon=depots_lon,
                mode='markers',
                marker=Marker(
                    color=color,
                    # symbol='car',
                    size=8
                ),
                hoverinfo='none'
            )
            cluster_data.append(depots)
        except:
            pass

        core_location = cluster_dict[i]['core']
        core = Scattermapbox(
            lat=[core_location[1]],
            lon=[core_location[0]],
            mode='markers',
            marker=Marker(
                # color='rgb(255,255,255)',
                color=color,
                # symbol='star',
                size=12
            ),

        )
        cluster_data.append(core)

    data = Data(cluster_data)

    layout = Layout(
        title='Clustering Model of Shanghai delivery Locations',
        # autosize=True,
        width="2000",
        height="2000",
        hovermode='closest',
        showlegend=False,
        mapbox=dict(
            accesstoken=mapbox_access_token,
            bearing=0,
            center=dict(
                lat=31,
                lon=121
            ),
            pitch=0,
            zoom=10,
            style='light'
        ),
    )

    fig = dict(data=data, layout=layout)
    plotly.offline.plot(fig, filename=save_name)

if __name__ == '__main__':
    delivery_cluster_num = 15
    size_weight = 0.8
    cover_weight = 1.2
    save_name = "{}_{}_{}_delivery_cluster_result.json".format(delivery_cluster_num, size_weight, cover_weight)
    html_name = "{}_{}_{}_delivery_cluster_result.html".format(delivery_cluster_num, size_weight, cover_weight)
    cluster_json = json.load(open(save_name))
    clusters_dict = dict((int(key), value) for key, value in cluster_json.items())
    map_visual(clusters_dict, cluster_n=delivery_cluster_num, delivery_points=True, save_name=html_name)
