# Import the required modules
import streamlit as st
import os
import pandas as pd
from pathlib import Path
from streamlit_option_menu import option_menu
import psycopg2
import requests
import numpy as np
import math
import random
from geopy.geocoders import Nominatim
from scripts import formulas

# Set the path to the project root and the folders for scripts and files
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

# Establish a connection to the database
conn = psycopg2.connect(
    host=db_host2,
    user=db_user2,
    password=db_password2,
    database=db_database2)

def drivers_pedido(name):
    # Add CSS style to change the font size, family, and color of the text displayed by the function
    st.markdown(""" <style> .font {
        font-size:40px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.write("Información de pedido asignado al repartidor")

    # Call an API to get data for available drivers and store it in a pandas dataframe
    df_repartidores = pd.DataFrame(formulas.get_repartidores_api())

    # get pedidos data from API and create a dataframe
    df_pedidos = pd.DataFrame(formulas.get_pedidos_api())
    
    '''
    The code checks if an input email exists in the 'email' column of the 
    'df_repartidores' dataframe, and if so, it returns the corresponding 'id' 
    value from the 'id' column of the same dataframe.
    '''
    # check if the email exists in the 'email' column of the DataFrame
    if name in df_repartidores['email'].values:
        # if the email exists, return the 'id' corresponding to that email
        repartidor_id = df_repartidores.loc[df_repartidores['email'] == name, 'id'].iloc[0]
        
    else:
        # if the email does not exist, return None
        pass
    
    col1, col2 = st.columns(2) # Create two columns 
    with col1:
        '''
        This code retrieves the "ocupado" value from the database for a given driver, which indicates 
        whether they are currently delivering or not. If the value is found, it displays a message 
        indicating whether they are currently delivering or not. If the value is not found, it displays 
        a message indicating that the user was not found.
        '''
        repartidor_pedidos = df_pedidos.loc[(df_pedidos["id_repartidor"] == repartidor_id) & 
                                    (df_pedidos["status"].isin(["Preparado", "En reparto"]))]


        # If the "ocupado" value is found, display whether the driver is currently delivering or not delivering
        if not repartidor_pedidos.empty:
            st.markdown('<h1 style="color: green;">Estás repartiendo</h1>', unsafe_allow_html=True)
        else:
            st.warning("No estás repartiendo")
   
    # Retrieve the status of the driver with the specified username(the username for login is the email)
    username = name 
    
    # Establish a database connection and create a cursor object
    cursor = conn.cursor()
    
    # Define the column layout using beta_columns()    
    col1, col2 = st.columns(2) # Create two columns 
    with col1:
        st.write("Cambia tu estado a entregado")
        '''
        This code block checks if the "No Repartiendo" button has been clicked. If it has been clicked, 
        the code updates the "ocupado" value in the database to False for the specified email. Then it 
        displays a message to confirm that the "ocupado" value has been updated.
        '''
        # If the "No Repartiendo" button is clicked, update the "ocupado" value to False
        if st.button("Entregado"):
            # First, retrieve the current value of "ocupado" for the specified email
            cursor.execute("SELECT status FROM repartidores WHERE email = %s", (username,))
            result = cursor.fetchone()

            # If the "status" value is found, check if True or False
            if result is not None:
                status = result[0]

                # If False, the user is not working, so he cannot update "ocupado" status
                if status == False:
                    st.info("No estás en horario laboral, no puedes cambiar tu estado")
                # If True, the user is working, so he can update "ocupado" status    
                else:                  
                    # Update the "ocupado" value in the database for the specified email
                    cursor.execute("UPDATE repartidores SET ocupado = %s WHERE email = %s", (False, username))
                    # Update the "status" value in the "repartidores" table to False for the driver with the specified username
                    selected_option = "Entregado"
                    repartidor_id = int(repartidor_id)
                    cursor.execute("UPDATE pedidos SET status = %s WHERE id_repartidor = %s AND status != 'Entregado'", (selected_option, repartidor_id))
                    conn.commit()
                 
                    # Display a message to confirm that the "ocupado" value has been updated
                    st.info("El estado 'ocupado' ha sido actualizado a 'No Repartiendo'")
                    # If button is clicked, rerun the Streamlit app
                    st.experimental_rerun()

    repartidor_pedidos = df_pedidos.loc[(df_pedidos["id_repartidor"] == repartidor_id) & 
                                (df_pedidos["status"].isin(["Preparado", "En reparto"]))]


    # If the "ocupado" value is found details of order.
    if not repartidor_pedidos.empty:
    
        '''
        Información sobre el pedido
        ''' 
        # The following line writes a string to the Streamlit app displaying some information about a pedido.
        st.write("Esta es la información del pedido")

        # The following line filters the pedidos dataframe for a specific repartidor_id and specific pedido status.
        repartidor_pedidos = df_pedidos[(df_pedidos["id_repartidor"] == repartidor_id) & (df_pedidos["status"].isin(["Preparado", "En reparto"]))]

        # The following line selects only two columns from the filtered dataframe, namely 'id_pedido' and 'datetime_pedido'.
        repartidor_pedidos_filtered = repartidor_pedidos[['id_pedido', 'datetime_pedido']]

        # The following line renames the columns of the filtered dataframe to 'ID Pedido' and 'Fecha Pedido', respectively.
        repartidor_pedidos_filtered = repartidor_pedidos_filtered.rename(columns={'id_pedido': 'ID Pedido', 'datetime_pedido': 'Fecha Pedido'})

        # The following line displays the filtered and renamed dataframe with 'ID Pedido' as the index.
        st.dataframe(repartidor_pedidos_filtered.set_index('ID Pedido'))
        
 
        '''
        Información sobre el lugar de recogida del pedido
        '''
        
        # The following line gets the 'id_comercio' value from the 'repartidor_pedidos' dataframe and converts it to an integer.
        idcomer = int(repartidor_pedidos["id_comercio"].values)

        # The following line retrieves the comercios data from an API and creates a dataframe.
        df_comercios = pd.DataFrame(formulas.get_comercios_api())

        # The following line filters the 'df_comercios' dataframe to include only the row that matches the 'idcomer' value.
        df_comercios = df_comercios[df_comercios['id'] == idcomer] 

        # The following line selects only two columns from the filtered 'df_comercios' dataframe, namely 'nombre' and 'direccion'.
        df_comercios_filtered = df_comercios[['nombre','direccion']]

        # The following line renames the columns of 'df_comercios_filtered' to 'Nombre' and 'Dirección', respectively.
        df_comercios_filtered = df_comercios_filtered.rename(columns={'nombre': 'Nombre', 'direccion': 'Dirección'})

        # The following line gets the 'longitud' value from the 'repartidor_pedidos' dataframe and converts it to a float.
        origen_long = float(repartidor_pedidos["longitud"].values)

        # The following line gets the 'latitud' value from the 'repartidor_pedidos' dataframe and converts it to a float.
        origen_lat = float(repartidor_pedidos["latitud"].values)

        # The following line gets the 'longitud' value for the 'idcomer' from the 'df_comercios' dataframe and converts it to a float.
        destin_long = float(df_comercios[df_comercios['id'] == idcomer]["longitud"].values)

        # The following line gets the 'latitud' value for the 'idcomer' from the 'df_comercios' dataframe and converts it to a float.
        destin_lat = float(df_comercios[df_comercios['id'] == idcomer]["latitud"].values)

        # The following line writes a string to the Streamlit app displaying information about the comercio where the pedido must be collected from.
        st.write("Esta es la información del comercio donde debes recoger el pedido")

        # The following line displays the time and distance between the repartidor and the comercio using the 'get_time_distance' function.
        st.write(formulas.get_time_distance(origen_long, origen_lat, destin_long, destin_lat,"tienda"))

        # The following line displays the 'df_comercios_filtered' dataframe with 'Nombre' as the index.
        st.dataframe(df_comercios_filtered.set_index('Nombre'))

        
        




        '''
        Información sobre el lugar de Entrega del pedido
        ''' 
        # Get the 'id_cliente' column from the 'repartidor_pedidos' dataframe and extract the values
        id_cliente_values = repartidor_pedidos['id_cliente'].values

        # Convert the extracted values to an integer data type
        id_cliente_values = int(id_cliente_values)

        # Retrieve a dataframe of all the clients using the 'get_clientes_api' function from the 'formulas' module
        df_clientes = pd.DataFrame(formulas.get_clientes_api())

        # Filter the 'df_clientes' dataframe to only include the rows where the 'id' column matches the 'id_cliente_values' variable
        df_clientes = df_clientes[df_clientes['id'] == id_cliente_values]

        # Filter the 'df_clientes' dataframe to only include the 'nombre' and 'direccion' columns
        df_clientes_filtered = df_clientes[['nombre','direccion']]

        # Rename the 'nombre' and 'direccion' columns of the 'df_clientes_filtered' dataframe to 'Nombre' and 'Dirección', respectively
        df_clientes_filtered = df_clientes_filtered.rename(columns={'nombre': 'Nombre', 'direccion': 'Dirección'})

        # Extract the longitude and latitude values of the delivery location ('destin') and the pickup location ('origen') and convert them to float data types
        origen_long2 = float(repartidor_pedidos["longitud"].values)
        origen_lat2 = float(repartidor_pedidos["latitud"].values)
        destin_long2 = float(df_clientes["longitud"].values)
        destin_lat2 = float(df_clientes["latitud"].values)

        st.write("Esta es la información del cliente")
        #st.write(formulas.get_time_distance(origen_long2, origen_lat2, destin_long2, destin_lat2,"cliente")) 
        st.dataframe(df_clientes_filtered.set_index('Nombre'))
        # Close the cursor
        cursor.close()
    else:
        pass