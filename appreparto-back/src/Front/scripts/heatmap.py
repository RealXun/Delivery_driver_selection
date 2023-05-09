import streamlit as st
import os
from pathlib import Path
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium, folium_static
from folium.plugins import MarkerCluster
import json
from streamlit_option_menu import option_menu
import pandas as pd
import requests
from scripts import formulas


PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
scripts_folder = (PROJECT_ROOT + "/" + "scripts")
files_folder = (scripts_folder + "/" + "files")
    
    
'''
This function creates a heatmap of the demand for goods by zones using real-time 
data obtained from two different API calls. It first creates dataframes from the 
two API calls, merges them by matching the IDs, extracts the latitude and longitude 
columns, creates a map centered on the mean latitude and longitude, creates a heatmap 
layer, adds the heatmap layer to the map, and displays the map using Streamlit.
'''
def heatmap_info(name):
    #Add the cover image for the cover page. Used a little trick to center the image
             # To display the header text using css style

    st.markdown(""" <style> .font {
        font-size:40px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.write("Mapa de calor en tiempo real de la demanda por zonas")
    
    # get comercios data from API and create a dataframe
    df_comercios = pd.DataFrame(formulas.get_comercios_api())

    # get pedidos data from API and create a dataframe
    df_pedidos = pd.DataFrame(formulas.get_pedidos_api())

    # join the comercios and pedidos dataframes by matching the ids
    df_merged = pd.merge(df_comercios, df_pedidos, left_on='id', right_on='id_comercio')
    st.write()
    # extract the latitude and longitude columns from the merged dataframe
    latitudes = df_merged['latitud_x']
    longitudes = df_merged['longitud_x']

    # create a Folium map centered on the mean latitud and longitud of the data
    map = folium.Map(location=[latitudes.mean(), longitudes.mean()], zoom_start=12, control_scale=True)

    # create the heatmap layer using latitudes and longitudes
    heatmap_layer = HeatMap(zip(latitudes, longitudes), min_opacity=0.2)

    # add the heatmap layer to the map
    heatmap_layer.add_to(map)

    # display the map in Streamlit
    folium_static(map)


