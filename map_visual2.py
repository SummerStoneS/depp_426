
import plotly
from plotly.graph_objs import *
import random


# mapbox_access_token = 'pk.eyJ1Ijoic3VtbWVyc3RvbmVzIiwiYSI6ImNqZmFrMmZzdDBhcXUyem1zdWl2N20ycWsifQ.46i6Xy2HDRhBArwqWGUd6Q'
mapbox_access_token = 'pk.eyJ1Ijoic25vd3dhbGtlcmoiLCJhIjoiY2piNGgxOWwyMnNvbzMyczc5aXgwbTY5dCJ9.swOGp-XV74Eh0Ow3ErtEXQ'


def map_visual(df, cluster_n=26, centers=None, cars=None, save_name='Shanghai pickup locations'):
    cluster_data = []
    for i in range(cluster_n):
        site_lon = df[df['cluster'] == i]['经度']
        site_lat = df[df['cluster'] == i]['纬度']
        color = 'rgb'+str((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
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
    if centers:
        centre = Scattermapbox(
            lat=centers[1],
            lon=centers[0],
            mode='markers',
            marker=Marker(
                color='rgb(255,0,0)',
                symbol='star',
                size=8
            ),
            text=["center"]*len(centers),
        )
        cluster_data.append(centre)

    if cars:
        car = Scattermapbox(
            lat=cars[1],
            lon=cars[0],
            mode='markers',
            marker=Marker(
                color='rgb(255,255,255)',
                symbol='car',
                size=5
            ),

        )
        cluster_data.append(car)

    data = Data(cluster_data)

    layout = Layout(
        title='Clustering Model of Shanghai Pickup Locations',
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

# map_visual(df, cluster_n=26)

