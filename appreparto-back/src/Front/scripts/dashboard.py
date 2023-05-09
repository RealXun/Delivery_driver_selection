import streamlit as st
import os
from pathlib import Path
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium, folium_static
import json
from streamlit_option_menu import option_menu
from scripts import shops
from streamlit_option_menu import option_menu
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
scripts_folder = (PROJECT_ROOT + "/" + "scripts")
files_folder = (scripts_folder + "/" + "files")


def kpis(name):
    #Add the cover image for the cover page. Used a little trick to center the image
             # To display the header text using css style

    st.title('**KPIS Menu**')
    st.markdown(""" <style> .font {
        font-size:40px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.write("We can see in real time the HOT (more busy) areas")

    ### This json file emulates the iformation we should be getting from the back
    ## el esquena de datos debe ser el siguiente :
    # [{'latitude': 40.415106, 'longitude': -3.712051}, {'latitude': 40.417227, 'longitude': -3.711617}]
    # Load the JSON file into a dictionary
    with open('my_dict_heatmap.json', 'r') as f:
        my_dict = json.load(f)
    # Print the resulting dictionary
    print(my_dict)


    ## el esquena de datos debe ser el siguiente :
    # [{'latitude': 40.415106, 'longitude': -3.712051}, {'latitude': 40.417227, 'longitude': -3.711617}]
    ## Once we get the information we use this
    # Create an empty list to store coordinates
    coordinates = []
    # Extract latitude and longitude data from each dictionary and add to coordinates list
    for item in my_dict:
        coordinates.append([item['latitude'], item['longitude']])

    # Create map
    m = folium.Map(location=[40.4168, -3.7038], zoom_start=15)
    # Add a heatmap layer to the map using the coordinates list
    HeatMap(coordinates).add_to(m)
    # Display map
    folium_static(m)