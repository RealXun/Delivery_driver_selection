import streamlit as st
import os
from pathlib import Path
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium, folium_static
import json
from streamlit_option_menu import option_menu
from scripts import delivering_drives
from scripts import available_drives

PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
scripts_folder = (PROJECT_ROOT + "/" + "scripts")
files_folder = (scripts_folder + "/" + "files")




def drivers_status(name):

    st.title('**Mapas de localización de los repartidores y su estado**')
    
    # Set a custom font style for the title
    st.markdown(""" <style> .font {
        font-size:40px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)

    # Create a dropdown menu to select the type of drivers to display
    selected2 = option_menu(
        None, ["Repartidores en reparto", "Repartidores no repartiendo"], 
        icons=['123', 'list-ul', 'credit-card-2-front'], 
        menu_icon="cast", 
        default_index=0, 
        orientation="horizontal")

    # Depending on the selected option, call a different function to display the drivers' map
    if selected2 == "Repartidores en reparto":
        delivering_drives.delivering_driv(name)

    elif selected2 == "Repartidores no repartiendo":
        available_drives.available_drives(name)
