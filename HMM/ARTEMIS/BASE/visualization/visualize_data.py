import branca.colormap as cmap
import folium
from folium.plugins import TimeSliderChoropleth
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import multivariate_normal


def load_data():
    # load the corona data
    corona_df = pd.read_csv('data/coronadata_Munich.csv')
    munich_map_df = gpd.read_file('data/shape/vg2500_krs.shp')
    return corona_df, munich_map_df


def extract_data(dataframe):
    # We only need the GEN and geometry columns
    munich_map_df = dataframe[['GEN', 'geometry']]

    # Only keep Munich and LK Munich
    munich_map_df = munich_map_df.loc[[224, 239]]
    munich_map_df['GEN'] = ['Munich', 'LK Munich']
    return munich_map_df


def join_dataframes(corona_df, munich_map_df):
    # Both dataframes have the column "GEN", we merge them using this column
    joined_df = corona_df.merge(munich_map_df, on='GEN')

    # The ObservationDate is given in date and time, we convert it to unix time in nanoseconds
    joined_df['date_sec'] = pd.to_datetime(joined_df['ObservationDate']).astype(np.int64) / 10 ** 9
    joined_df['date_sec'] = joined_df['date_sec'].astype(int).astype(str)

    # Delete the ObservationDate column as we do not need it anymore
    joined_df = joined_df.drop('ObservationDate', axis=1)
    return joined_df


def visualize_corona_data(joined_df, mode='Confirmed'):
    # Here, we visualize the daily confirmed case
    # Add the color to each row
    max_colour = max(joined_df[mode])
    min_colour = min(joined_df[mode])
    colour_map = cmap.linear.YlOrRd_09.scale(min_colour, max_colour)
    joined_df['colour'] = joined_df[mode].map(colour_map)

    # create an inner dictionary for the visualization
    geo_list = joined_df['GEN'].unique().tolist()
    geo_idx = range(len(geo_list))

    style_dict = {}
    for i in geo_idx:
        geo = geo_list[i]
        result = joined_df[joined_df['GEN'] == geo]
        inner_dict = {}
        for _, r in result.iterrows():
            inner_dict[r['date_sec']] = {'color': r['colour'], 'opacity': 0.7}
        style_dict[str(i)] = inner_dict

    # create a geo_gdf for the visualization
    geo_df = joined_df[['geometry']]
    geo_gdf = gpd.GeoDataFrame(geo_df)
    geo_gdf = geo_gdf.drop_duplicates().reset_index()

    # You might need to change the value of min_zoom depending on your platform
    slider_map = folium.Map(location=[48.08, 11.61], min_zoom=2, max_bounds=True)

    _ = TimeSliderChoropleth(data=geo_gdf.to_json(), styledict=style_dict).add_to(slider_map)
    _ = colour_map.add_to(slider_map)

    if mode == 'Confirmed':
        colour_map.caption = "Confirmed cases in the past 7 days per 100,000 people"
    else:
        colour_map.caption = "Deaths in the past 7 days per 5,000,000 people"
    return colour_map, slider_map


def vis_plt(trajectory, description, months):
    plt.figure(figsize=(20, 20))
    plt.subplot(2, 1, 1)
    fig = plt.step(months, trajectory, color="#8dd3c7", where="pre", lw=2)
    plt.ylim(1, 7)
    plt.title(description)
    return fig


def vis_dataframe(data):
    df = pd.DataFrame(np.transpose(data), columns=['Months', 'Munich', 'LK_Munich'])
    return df
