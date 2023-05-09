import streamlit as st
import os
from pathlib import Path
import psycopg2
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
def delivering_driv(name):
    #Add the cover image for the cover page. Used a little trick to center the image
             # To display the header text using css style

    st.markdown(""" <style> .font {
        font-size:40px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.write("Mapa de los repartidores que están en horario laboral y repartiendo")

    '''
    This code gets data for available drivers from an API and stores it in a pandas dataframe. 
    It then filters this dataframe to only include drivers who are currently available and delivering, 
    creating a new dataframe. The resulting dataframe contains data for available drivers, 
    which can be used to create a map of their locations.
    '''

    # Call an API to get data for available drivers and store it in a pandas dataframe
    df_repartidores = pd.DataFrame(formulas.get_repartidores_api())

    # Filter the dataframe to only include available drivers who are currently working
    available_drivers = df_repartidores[(df_repartidores['status'] == True) & (df_repartidores['ocupado'] == True)]



    '''
    This code creates a map using the Folium library and centers it on the mean latitude and longitude 
    of the available drivers. It creates a marker cluster to group markers together for better readability 
    on the map. It then loops through each available driver and creates a marker for them on the map. 
    For each driver, it creates a popup that displays information about the driver and sets its dimensions. 
    It then creates a marker for the driver and sets its location, popup, and icon based on the vehicle 
    type. Finally, it adds the marker to the marker cluster.
    '''
    
    # Create a folium map centered on the mean latitude and longitude of the available drivers
    m = folium.Map(location=[available_drivers.latitud.mean(), available_drivers.longitud.mean()], zoom_start=12, control_scale=True)

    # Create a marker cluster to group markers together for better readability on the map
    marker_cluster = MarkerCluster().add_to(m)

    # Loop through each available driver and create a marker for them on the map
    for i, row in available_drivers.iterrows():
        
        # Create a popup for the marker that displays information about the driver
        popup_html = f"<b>Driver ID:</b> {row['id']}<br>\
        <b>Nombre:</b> {row['nombre']}<br>\
        <b>Tipo de Vehículo:</b> {row['vehiculo']}<br>\
        <b>Trabajando:</b> {row['status']}<br>\
        <b>Repartiendo:</b> {row['ocupado']}<br>\
        <b>E-mail:</b>{row['email']}<br>"
        
        # Create an iframe to hold the popup and set its dimensions
        iframe_html = folium.IFrame(popup_html, width=250, height=180)
        
        # Create the popup and set a max width
        popup = folium.Popup(iframe_html, max_width=2500)
        
        # Create a marker for the driver and set its location, popup, and icon based on the vehicle type
        if  row['vehiculo'] == "Motocicleta":
            marker = folium.Marker(location=[row['latitud'], row['longitud']], popup=popup, icon=folium.Icon(color='red', icon='motorcycle', prefix='fa'))
        elif row['vehiculo'] == "Bicicleta":
            marker = folium.Marker(location=[row['latitud'], row['longitud']], popup=popup, icon=folium.Icon(color='red', icon="fa-thin fa-bicycle", prefix='fa'))
        else:
            marker = folium.Marker(location=[row['latitud'], row['longitud']], popup=popup, icon=folium.Icon(color='red', icon="fa-light fa-charging-station", prefix='fa'))
        
        # Add the marker to the marker cluster
        marker_cluster.add_child(marker)

    # Display the folium map in the streamlit app
    folium_static(m)