import streamlit as st
import os
from pathlib import Path
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium, folium_static
import json
from streamlit_option_menu import option_menu
from scripts import pedido_reparto
from scripts import pedido_entregado
from scripts import pedido_preparado
from scripts import pedido_recibido
from scripts import formulas
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
scripts_folder = (PROJECT_ROOT + "/" + "scripts")
files_folder = (scripts_folder + "/" + "files")




def pedidos(name):
    #Add the cover image for the cover page. Used a little trick to center the image
             # To display the header text using css style
    st.title('**Estado global de los pedidos**')
    st.markdown(""" <style> .font {
        font-size:40px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)




    selected2 = option_menu(
        None, ["Recibidos","Preparados","En reparto", "Entregados"], 
        icons=['123', 'list-ul', 'credit-card-2-front'], 
        menu_icon="cast", 
        default_index=0, 
        orientation="horizontal")
        
    if selected2 == "Recibidos":
        pedido_recibido.pedido_recibido(name)

    elif selected2 == "Preparados":
        pedido_preparado.pedido_preparado(name)
        
    elif selected2 == "En reparto":
        pedido_reparto.pedido_reparto(name)

    elif selected2 == "Entregados":
        pedido_entregado.pedido_entregado(name)