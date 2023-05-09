import streamlit as st
import os
from pathlib import Path
import psycopg2
import pandas as pd
from streamlit_option_menu import option_menu
from datetime import datetime
import random
import requests
from sklearn.neighbors import NearestNeighbors
import numpy as np
from scripts import formulas

PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
scripts_folder = (PROJECT_ROOT + "/" + "scripts")
files_folder = (scripts_folder + "/" + "files")

# Get database credentials from environment variables
db_host2 = os.environ.get('hostnew')
db_user2 = os.environ.get('user')
db_password2 = os.environ.get('password')
db_database2 = os.environ.get('database')


'''
Main Body
'''
def pedidos_tienda(name,id):
    
    #Add the cover image for the cover page. Used a little trick to center the image
             # To display the header text using css style

    st.title('**Menú tienda donde ve el estado de sus pedidos**')
    st.markdown(""" <style> .font {
        font-size:40px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    
    # Define the column layout using beta_columns()    
    col1, col2, col3, col4 = st.columns(4) # Create two columns
    with col1:
        '''
        Codigo donde muestras los pedidos recibidos
        '''
        st.write("Pedidos recibidos")
        
        # Create a Pandas DataFrame from the data retrieved from the API
        df_pedidos_recibidos = pd.DataFrame(formulas.get_pedidos_api())

        # Get the id_comercio to filter by
        id_comercio = id

        # Filter the DataFrame by the specified id_comercio
        df_pedidos_recibidos = df_pedidos_recibidos[df_pedidos_recibidos['id_comercio'] == id_comercio]

        # Filter the DataFrame to include only orders with status "Recibido"
        df_pedidos_recibidos = df_pedidos_recibidos[df_pedidos_recibidos['status'] == "Recibido"]

        # Select only the columns "id_pedido", "status", and "datetime_pedido"
        df_pedidos_recibidos = df_pedidos_recibidos[['id_pedido', 'status', 'datetime_pedido']]

        # Rename the selected columns to "ID Pedido", "Estado", and "Fecha Pedido"
        df_pedidos_recibidos = df_pedidos_recibidos.rename(columns={'id_pedido': 'ID Pedido', 'status': 'Estado', 'datetime_pedido': 'Fecha'})

        # Sort the DataFrame by "ID Pedido" in ascending order
        df_pedidos_recibidos = df_pedidos_recibidos.sort_values(by=['ID Pedido'], ascending=True)

        # Reset the DataFrame index to start from 0
        df_pedidos_recibidos = df_pedidos_recibidos.reset_index(drop=True)

        # Display the DataFrame in Streamlit using the set_index() method to set the index to "ID Pedido"
        st.dataframe(df_pedidos_recibidos.set_index('ID Pedido'))

 
    with col2: 
        '''
        Codigo donde muestras los pedidos preparados
        '''
        st.write("Preparados")
        
        # Create a Pandas DataFrame from the data retrieved from the API
        df_pedidos_preparado = pd.DataFrame(formulas.get_pedidos_api())

        # Get the id_comercio to filter by
        id_comercio = id

        # Filter the DataFrame by the specified id_comercio
        df_pedidos_preparado = df_pedidos_preparado[df_pedidos_preparado['id_comercio'] == id_comercio]

        # Filter the DataFrame to include only orders with status "Recibido"
        df_pedidos_preparado = df_pedidos_preparado[df_pedidos_preparado['status'] == "Preparado"]

        # Select only the columns "id_pedido", "status", and "datetime_pedido"
        df_pedidos_preparado = df_pedidos_preparado[['id_pedido','id_repartidor', 'status', 'datetime_pedido']]

        # Rename the selected columns to "ID Pedido", "Estado", and "Fecha Pedido"
        df_pedidos_preparado = df_pedidos_preparado.rename(columns={'id_pedido': 'ID Pedido','id_repartidor':'ID Repartidor', 'status': 'Estado', 'datetime_pedido': 'Fecha'})

        # Sort the DataFrame by "ID Pedido" in ascending order
        df_pedidos_preparado = df_pedidos_preparado.sort_values(by=['ID Pedido'], ascending=True)

        # Reset the DataFrame index to start from 0
        df_pedidos_preparado = df_pedidos_preparado.reset_index(drop=True)

        # Display the DataFrame in Streamlit using the set_index() method to set the index to "ID Pedido"
        st.dataframe(df_pedidos_preparado.set_index('ID Pedido'))

    with col3:    
        '''
        Codigo donde muestras los pedidos En reparto
        '''
        st.write("En reparto")
        
        # Create a Pandas DataFrame from the data retrieved from the API
        df_pedidos_en_reparto = pd.DataFrame(formulas.get_pedidos_api())

        # Get the id_comercio to filter by
        id_comercio = id

        # Filter the DataFrame by the specified id_comercio
        df_pedidos_en_reparto = df_pedidos_en_reparto[df_pedidos_en_reparto['id_comercio'] == id_comercio]

        # Filter the DataFrame to include only orders with status "Recibido"
        df_pedidos_en_reparto = df_pedidos_en_reparto[df_pedidos_en_reparto['status'] == "En reparto"]

        # Select only the columns "id_pedido", "status", and "datetime_pedido"
        df_pedidos_en_reparto = df_pedidos_en_reparto[['id_pedido','id_repartidor', 'status', 'datetime_pedido']]

        # Rename the selected columns to "ID Pedido", "Estado", and "Fecha Pedido"
        df_pedidos_en_reparto = df_pedidos_en_reparto.rename(columns={'id_pedido': 'ID Pedido','id_repartidor':'ID Repartidor', 'status': 'Estado', 'datetime_pedido': 'Fecha'})

        # Sort the DataFrame by "ID Pedido" in ascending order
        df_pedidos_en_reparto = df_pedidos_en_reparto.sort_values(by=['ID Pedido'], ascending=True)

        # Reset the DataFrame index to start from 0
        df_pedidos_en_reparto = df_pedidos_en_reparto.reset_index(drop=True)

        # Display the DataFrame in Streamlit using the set_index() method to set the index to "ID Pedido"
        st.dataframe(df_pedidos_en_reparto.set_index('ID Pedido'))

    with col4:
        '''
        Codigo donde muestras los pedidos Entregados
        '''
        st.write("Entregados")
        
        # Create a Pandas DataFrame from the data retrieved from the API
        df_pedidos_entregado = pd.DataFrame(formulas.get_pedidos_api())

        # Get the id_comercio to filter by
        id_comercio = id

        # Filter the DataFrame by the specified id_comercio
        df_pedidos_entregado = df_pedidos_entregado[df_pedidos_entregado['id_comercio'] == id_comercio]

        # Filter the DataFrame to include only orders with status "Recibido"
        df_pedidos_entregado = df_pedidos_entregado[df_pedidos_entregado['status'] == "Entregado"]

        # Select only the columns "id_pedido", "status", and "datetime_pedido"
        df_pedidos_entregado = df_pedidos_entregado[['id_pedido', 'status', 'datetime_pedido']]

        # Rename the selected columns to "ID Pedido", "Estado", and "Fecha Pedido"
        df_pedidos_entregado = df_pedidos_entregado.rename(columns={'id_pedido': 'ID Pedido', 'status': 'Estado', 'datetime_pedido': 'Fecha'})

        # Sort the DataFrame by "ID Pedido" in ascending order
        df_pedidos_entregado = df_pedidos_entregado.sort_values(by=['ID Pedido'], ascending=True)

        # Reset the DataFrame index to start from 0
        df_pedidos_entregado = df_pedidos_entregado.reset_index(drop=True)

        # Display the DataFrame in Streamlit using the set_index() method to set the index to "ID Pedido"
        st.dataframe(df_pedidos_entregado.set_index('ID Pedido'))


    '''
    Codigo donde la tienda mete la informacion del pedido a cambiar
    '''
    # Prompts the user to enter an oder id 
    # get and stores the input in the 'user_input' variable.    
    user_input = st.text_input("¿Que id de pedido quiere modificar?")
    
    # The input is then converted into an integer type and stored in the 
    # 'id_chosen' variable using the 'int' function. 
    try:
        id_chosen = int(user_input)
    except ValueError:
        st.error('Completa el campo con un numero entero para poder realizar la acción')
        
        
    option = ["Preparado", "En reparto","Entregado"]
    selected_option = st.multiselect('Selecciona opción de estado del pedido', option, max_selections=1) 
    selected_option = ', '.join(selected_option)
    
    '''
    Esta parte crea un botón para modificar el estado de un pedido de forma manual
    '''

    criteria_selected = user_input and selected_option

    # Create a visual indicator to show if both criteria are selected
    if criteria_selected:
        st.success('Todos los criterios están seleccionados. Puede hacer clic ahora.')
    else:
        st.warning('Completa los campos para poder realizar la acción')
        
    if st.button('Cambiar el estado', disabled=not criteria_selected):
        with st.spinner('Cambiando el estado......'):
            if selected_option == "Preparado":
                # Establish a connection to the database
                conn = psycopg2.connect(
                    host=db_host2,
                    user=db_user2,
                    password=db_password2,
                    database=db_database2)
                # Establish a database connection and create a cursor object
                cursor = conn.cursor()    
                '''
                We called the funciton to finde the closest driver
                '''
                # Function to get the ID of the closes driver
                id_repartidor_closest = int(formulas.find_driver(id))
                # Update the "status" value in the "pedidos" table to False for the driver with the specified username
                cursor.execute("UPDATE pedidos SET status = %s , id_repartidor = %s  WHERE id_pedido = %s", (selected_option,id_repartidor_closest, id_chosen))
                 # Update the "ocupado" value in the database for the specified email
                cursor.execute("UPDATE repartidores SET ocupado = %s WHERE id = %s", (True, id_repartidor_closest))
                # Commit the changes to the database
                conn.commit()
            elif selected_option == "En reparto":
                # Establish a connection to the database
                conn = psycopg2.connect(
                    host=db_host2,
                    user=db_user2,
                    password=db_password2,
                    database=db_database2)
                # Establish a database connection and create a cursor object
                cursor = conn.cursor()  
                # Update the "status" value in the "repartidores" table to False for the driver with the specified username
                cursor.execute("UPDATE pedidos SET status = %s WHERE id_pedido = %s", (selected_option, id_chosen))                
               
            else :
                # Establish a connection to the database
                conn = psycopg2.connect(
                    host=db_host2,
                    user=db_user2,
                    password=db_password2,
                    database=db_database2)
                # Establish a database connection and create a cursor object
                cursor = conn.cursor()  
                df_ped = pd.DataFrame(formulas.get_pedidos_api())
                # Filter the DataFrame to keep only the row(s) where id_pedido matches the given value
                filtered_df = df_ped[df_ped["id_pedido"] == id_chosen]
                # Extract the value(s) of id_comercio from the filtered DataFrame
                id_repartidor_values = filtered_df["id_repartidor"].values    
                id_repartidor_values = int(id_repartidor_values)               
                # Update the "status" value in the "repartidores" table to False for the driver with the specified username
                cursor.execute("UPDATE pedidos SET status = %s WHERE id_pedido = %s", (selected_option, id_chosen))
                # Update the "ocupado" value in the database for the specified email
                cursor.execute("UPDATE repartidores SET ocupado = %s WHERE id = %s", (False, id_repartidor_values))                

            # Commit the changes to the database
            conn.commit()
            # Close the cursor
            cursor.close()
            # Close the connection
            conn.close()

            # Display a message to indicate that the status has been updated
            st.write("Status updated")

            # If button is clicked, rerun the Streamlit app
            st.experimental_rerun()   