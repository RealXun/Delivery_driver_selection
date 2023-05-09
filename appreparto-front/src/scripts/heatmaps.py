import streamlit as st
import os
from pathlib import Path
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium, folium_static
import json
from streamlit_option_menu import option_menu
from scripts import heatmap
from scripts import heatmap_predict

PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
scripts_folder = (PROJECT_ROOT + "/" + "scripts")
files_folder = (scripts_folder + "/" + "files")




def heatmaps(name):
    #Add the cover image for the cover page. Used a little trick to center the image
             # To display the header text using css style

    st.title('**Mapas de calor de los pedidos**')
    st.markdown(""" <style> .font {
        font-size:40px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)




    selected2 = option_menu(
        None, ["En Tiempo Real", "Predicción"], 
        icons=['123', 'list-ul', 'credit-card-2-front'], 
        menu_icon="cast", 
        default_index=0, 
        orientation="horizontal")

    if selected2 == "En Tiempo Real":
        heatmap.heatmap_info(name)

    elif selected2 == "Predicción":
        heatmap_predict.heatmap_predict(name)