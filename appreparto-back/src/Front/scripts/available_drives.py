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
Main body of the page
'''
def available_drives(name):
    #Add the cover image for the cover page. Used a little trick to center the image
             # To display the header text using css style
    st.markdown(""" <style> .font {
        font-size:40px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.write("Mapa de los repartidores que están en horario laboral pero no repartiendo")

    '''
    This code gets data for available drivers from an API and stores it in a pandas dataframe. 
    It then filters this dataframe to only include drivers who are currently available and not delivering, 
    creating a new dataframe. The resulting dataframe contains data for available drivers, 
    which can be used to create a map of their locations.
    '''
    # Create a pandas dataframe using data obtained from an API call to get driver information
    df_repartidores = pd.DataFrame(formulas.get_repartidores_api())

    # Filter the dataframe to only include drivers who are available and not currently working
    available_drivers = df_repartidores[(df_repartidores['status'] == True) & (df_repartidores['ocupado'] != True)]


    '''
    This code generates a map using the Folium library, showing the location of available drivers. 
    It uses data from an API to create a Pandas DataFrame with information about the drivers. 
    The map is centered on the mean location of the available drivers and includes markers for each 
    driver's location. The marker icons are colored based on the type of vehicle and include 
    information about the driver in a popup when clicked. Finally, the map is displayed using 
    Streamlit's folium_static function.
    '''
    # Create a map centered on the mean latitude and longitude of available drivers with zoom level 12
    m = folium.Map(location=[available_drivers.latitud.mean(), available_drivers.longitud.mean()], zoom_start=12, control_scale=True)

    # Create a MarkerCluster to group markers together
    marker_cluster = MarkerCluster().add_to(m)

    # Loop through each available driver and create a marker with a popup containing driver information
    for i, row in available_drivers.iterrows():
        
        # Create a popup HTML string with driver information
        popup_html = f"<b>Driver ID:</b> {row['id']}<br>\
            <b>Nombre:</b> {row['nombre']}<br>\
            <b>Tipo de Vehículo:</b> {row['vehiculo']}<br>\
            <b>Trabajando:</b> {row['status']}<br>\
            <b>En reparto:</b> {row['status']}<br>\
            <b>E-mail:</b>{row['email']}<br>"
            
        # Create an IFrame to hold the popup HTML
        iframe_html = folium.IFrame(popup_html, width=250, height=180)
        
        # Create a Popup object using the IFrame
        popup = folium.Popup(iframe_html, max_width=2500)
        
        # Create a marker and add it to the MarkerCluster
        if  row['vehiculo'] == "Motocicleta":
            marker = folium.Marker(location=[row['latitud'], row['longitud']], popup=popup, icon=folium.Icon(color='green', icon='motorcycle', prefix='fa'))
        elif row['vehiculo'] == "Bicicleta":
            marker = folium.Marker(location=[row['latitud'], row['longitud']], popup=popup, icon=folium.Icon(color='green', icon="fa-thin fa-bicycle", prefix='fa'))
        else:
            marker = folium.Marker(location=[row['latitud'], row['longitud']], popup=popup, icon=folium.Icon(color='green', icon="fa-light fa-charging-station", prefix='fa'))
            
        # Add the marker to the marker cluster
        marker_cluster.add_child(marker)

    # Display the map
    folium_static(m)