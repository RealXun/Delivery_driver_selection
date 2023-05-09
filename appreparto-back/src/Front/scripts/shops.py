import streamlit as st
import os
from pathlib import Path
import pandas as pd
from streamlit_option_menu import option_menu
import requests
import folium
from streamlit_folium import st_folium, folium_static
from folium.plugins import MarkerCluster
from scripts import formulas

PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
scripts_folder = (PROJECT_ROOT + "/" + "scripts")
files_folder = (scripts_folder + "/" + "files")


'''
Main Body
'''
def shops_info(name):
    #Add the cover image for the cover page. Used a little trick to center the image
             # To display the header text using css style

    st.title('**Mapa con la localización de las tiendas**')
    st.markdown(""" <style> .font {
        font-size:40px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)

    df_comercios = pd.DataFrame(formulas.get_comercios_api())


    # Create a map centered on the mean latitude and longitude of shops with zoom level 12
    m = folium.Map(location=[df_comercios.latitud.mean(), df_comercios.longitud.mean()], zoom_start=12, control_scale=True)

    # Create a MarkerCluster to group markers together
    marker_cluster = MarkerCluster().add_to(m)
    
    # Loop through each available driver and create a marker with a popup containing driver information
    for i, row in df_comercios.iterrows():
        
        # Create a popup HTML string with driver information
        popup_html = f"<b>Driver ID:</b> {row['id']}<br>\
            <b>Nombre:</b> {row['nombre']}<br>\
            <b>Dirección:</b> {row['direccion']}<br>\
            <b>Códifo Postal:</b> {row['zona']}<br>\
            <b>Tipo:</b> {row['tipo']}<br>"
            
        # Create an IFrame to hold the popup HTML
        iframe_html = folium.IFrame(popup_html, width=250, height=180)
        
        # Create a Popup object using the IFrame
        popup = folium.Popup(iframe_html, max_width=2500)
        
        # Create a marker and add it to the MarkerCluster
        marker = folium.Marker(location=[row['latitud'], row['longitud']], popup=popup, icon=folium.Icon(color='green', icon='fa-light fa-utensils', prefix='fa'))
            
        # Add the marker to the marker cluster
        marker_cluster.add_child(marker)

    # Display the map
    folium_static(m)