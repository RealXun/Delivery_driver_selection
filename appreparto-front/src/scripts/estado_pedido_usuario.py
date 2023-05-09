import streamlit as st
import os
from pathlib import Path
import psycopg2
import pandas as pd
from streamlit_option_menu import option_menu
from datetime import datetime
import random
from scripts import formulas


PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
scripts_folder = (PROJECT_ROOT + "/" + "scripts")
files_folder = (scripts_folder + "/" + "files")

def estado_pedido_usuario(name):
    
    #Add the cover image for the cover page. Used a little trick to center the image
             # To display the header text using css style

    st.title('**Aquí puede ver el estado de sus pedidos**')
    st.markdown(""" <style> .font {
        font-size:40px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
  
    df_clientes = pd.DataFrame(formulas.get_clientes_api())
    '''
    Obtenemos el ID del cliente
    '''
    
    # Filter the DataFrame to keep only the row(s) where username matches the given value
    filtered_df = df_clientes[df_clientes["email"] == name]
    # Extract the value(s) of id_cliente from the filtered DataFrame
    id_cliente_values = filtered_df["id"].values
    # Check if id_cliente_values is not empty
    if id_cliente_values:
        # Convert id_cliente_values to an integer (assuming there's only one value)
        id_cliente = int(id_cliente_values[0])
        # Use id_cliente in your code as needed
    else:
        st.write(f"No matching id_cliente found for username {username}")
    
   
    

    '''
    Codigo donde muestras los pedidos 
    '''
    df_pedidos = pd.DataFrame(formulas.get_pedidos_api())
    # Filter the DataFrame to keep only the rows where id_cliente matches the given value
    filtered_df = df_pedidos[df_pedidos["id_cliente"] == id_cliente]
    st.write(filtered_df)
