import streamlit as st
import os
from pathlib import Path
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium, folium_static
import json
from streamlit_option_menu import option_menu
import psycopg2
import pandas as pd
from scripts import formulas

PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
scripts_folder = (PROJECT_ROOT + "/" + "scripts")
files_folder = (scripts_folder + "/" + "files")


def pedido_entregado(name):
    #Add the cover image for the cover page. Used a little trick to center the image
             # To display the header text using css style

    st.markdown(""" <style> .font {
        font-size:40px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.write("Pedidos entregados") 

    # Create a Pandas DataFrame from the data retrieved from the API
    df_pedidos_entregados = pd.DataFrame(formulas.get_pedidos_api())
    
    # Filter the DataFrame to include only orders with status "Recibido"
    df_pedidos_entregados = df_pedidos_entregados[df_pedidos_entregados['status'] == "Entregado"]
    
    # Rename the selected columns to "ID Pedido", "Estado", and "Fecha Pedido"
    df_pedidos_entregados = df_pedidos_entregados.rename(columns={'id_pedido': 'ID Pedido', 'id_cliente':'ID Cliente','id_comercio': 'ID Comercio', 'id_repartidor':'ID Repartidor','datetime_pedido': 'Fecha','status': 'Estado','tamaño':'Tamaño','direccion':'Direccion'})
    
    # Select only the columns "id_pedido", "status", and "datetime_pedido"
    df_pedidos_entregados = df_pedidos_entregados[['ID Pedido', 'ID Cliente', 'ID Comercio', 'ID Repartidor','Fecha', 'Estado', 'Tamaño', 'Direccion']]
    
    # Sort the DataFrame by "ID Pedido" in ascending order
    df_pedidos_entregados = df_pedidos_entregados.sort_values(by=['ID Pedido'], ascending=True)
    
    # Reset the DataFrame index to start from 0
    df_pedidos_entregados = df_pedidos_entregados.reset_index(drop=True)
    
    # Display the DataFrame in Streamlit using the set_index() method to set the index to "ID Pedido"# Establish a database connection and create a cursor object
    st.dataframe(df_pedidos_entregados.set_index('ID Pedido'))