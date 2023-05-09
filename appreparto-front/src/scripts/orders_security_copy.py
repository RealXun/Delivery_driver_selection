import streamlit as st
import os
from pathlib import Path
import psycopg2
import pandas as pd
from streamlit_option_menu import option_menu
from datetime import datetime
import random


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

def name_unique(df,name):
    '''
    Return a list of unique names of type of shops
    '''
    if name == "tipos":
        df = df[df["tipos"] != "restaurant"] # ESTOY HAY QUE ELIMINARLO UNA VEZ MODIFICADA LA BASE DE DATOS DE COMERCIOS
        df = df["tipos"].unique().tolist()
    else:
        df = df["nombre"].unique().tolist()   
    return df


def insert_oders(name):
    #Add the cover image for the cover page. Used a little trick to center the image
             # To display the header text using css style

    st.title('**We add orders**')
    st.markdown(""" <style> .font {
        font-size:40px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.write("We add orders and get a return about which driver will do it")


    '''
    Esta parte del código crea una lista desplegable de tipos de comercios
    '''
    # Establish a database connection and create a cursor object
    cursor = conn.cursor()
    
    # Execute the query
    cursor.execute("SELECT tipo FROM comercios")
    
    # We are going to fetch all the results from the query above
    result = cursor.fetchall()
    
    # Create a DataFrame from the results and specify the column names
    df = pd.DataFrame(result, columns = ["tipos"])


    # Define the list of unique values for the specified column
    type_options = sorted(name_unique(df, "tipos"))

    # Add a default option that is not one of the available values
    default_option = "Selecciona una opción"
    type_options.insert(0, default_option)

    # Create the selectbox with the list of options and the default value
    selection_type = st.selectbox("Elige el tipo de comida que quieres", type_options, index=0)
   
   
    '''
    Esta parte del código crea una lista desplegable de nombres de comercios a raíz del tipo seleccionado anteriormente
    '''   
    # Execute the query
    cursor.execute("SELECT nombre FROM comercios WHERE tipo = %s", (selection_type,))

    # We are going to fetch all the results from the query above
    result = cursor.fetchall()
    
    # Create a DataFrame from the results and specify the column names
    df = pd.DataFrame(result, columns = ["nombre"])
    
    # Creation of a selectbox that will display unique names of commerce
    selection_name = st.selectbox("Elige el restaurante",sorted(name_unique(df,"nombre")))



    '''
    Esta parte del código Recoge in input del pedido que quiere
    ''' 
    order_details = st.text_input("Escribe el pedido que deseas")



    '''
    Esta parte crea un botón de hacer pedido y actualiza la tabla de pedidos
    '''
    
    criteria_selected = selection_type and selection_name and order_details


    # Create a visual indicator to show if both criteria are selected
    if criteria_selected:
        st.success('All criteria are selected. You can click now.')
    else:
        st.warning('Completa los tres campos del pedido')
        
    if st.button('Hacer pedido', disabled=not criteria_selected):
        with st.spinner('Realizando pedido...'):
        
            '''
            Collecting information to do the order
            '''
            
            # Execute the query to fetch the latitud,longitud,direccion,id
            cursor.execute("SELECT latitud,longitud,direccion,id FROM clientes WHERE email = %s", (name,))
            datos_cliente = cursor.fetchall()
            # Divide the results into separate lists for each column
            latitudes, longitudes, direcciones, ids = zip(*datos_cliente)
            latitud = (", ".join(str(lat) for lat in latitudes))
            longitud = (", ".join(str(lat) for lat in longitudes))
            direccion = (", ".join(str(lat) for lat in direcciones))
            id_cliente = (", ".join(str(lat) for lat in ids))
            
            
            # Execute the query to fetch the shop id
            cursor.execute("SELECT id FROM comercios WHERE nombre = %s", (selection_name,))
            id_comercio = cursor.fetchall()

            # Getting datetime
            datetime_pedido = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            

            # Generate a random tamaño value between 0.2 and 2 kg
            tamaño = round(random.uniform(0.2, 2), 2)
            
            # Assign status as en reparto
            status = "En reparto"
            
            '''
            Update the pedidos table
            '''
           
            # Execute an SQL query to update the "status" column of the driver with the specified username to True
            cursor.execute("INSERT INTO pedidos (latitud, longitud, direccion, id_cliente, id_comercio, datetime_pedido, tamaño, status) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",(latitud, longitud, direccion, id_cliente, id_comercio[0], datetime_pedido, tamaño, status))
            # Commit the changes to the database
            conn.commit()
            # Close the cursor object
            cursor.close()
            # Close the database connection
            connection.close()
        st.write("Pedido realizado")
    # If button is clicked, rerun the Streamlit app
    st.experimental_rerun()
