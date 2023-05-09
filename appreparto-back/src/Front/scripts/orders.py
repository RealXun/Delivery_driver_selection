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

# Get database credentials from environment variables
db_host2 = os.environ.get('hostnew')
db_user2 = os.environ.get('user')
db_password2 = os.environ.get('password')
db_database2 = os.environ.get('database')

def insert_oders(name):
    
    #Add the cover image for the cover page. Used a little trick to center the image
             # To display the header text using css style

    st.title('**Realización de pedido**')
    st.markdown(""" <style> .font {
        font-size:40px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.write("Seleccione el tipo de comida que quiere, la tienda y escriba lo que quiere")

    df_comercios = pd.DataFrame(formulas.get_comercios_api())

    '''
    Esta parte del código crea una lista desplegable de tipos de comercios
    '''
    # Use the `unique` method to get a list of unique values in the "tipo" column
    unique_tipos = df_comercios["tipo"].unique().tolist()


    # Define the list of unique values for the specified column
    type_options = sorted(unique_tipos)

    # Add a default option that is not one of the available values
    default_option = "Selecciona una opción"
    type_options.insert(0, default_option)

    # Create the selectbox with the list of options and the default value
    selection_type = st.selectbox("Elige el tipo de comida que quieres", type_options, index=0)


   
    '''
    Esta parte del código crea una lista desplegable de nombres de comercios a raíz del tipo seleccionado anteriormente
    '''   
    # Filter the DataFrame to keep only rows where the "tipo" column matches the input tipo
    filtered_df = df_comercios[df_comercios["tipo"] == selection_type]

    # Use the `unique` method to get a list of unique "nombre" values in the filtered DataFrame
    unique_nombres = filtered_df["nombre"].unique().tolist()
    
    # Creation of a selectbox that will display unique names of commerce
    selection_name = st.selectbox("Elige el restaurante",sorted(unique_nombres))



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
        st.success('Todos los criterios están seleccionados. Puede hacer clic ahora.')
    else:
        st.warning('Completa los tres campos del pedido')
        
    if st.button('Hacer pedido', disabled=not criteria_selected):
        with st.spinner('Realizando pedido...'):
        
            '''
            Collecting information to do the order
            '''
            # Establish a connection to the database
            conn = psycopg2.connect(
                host=db_host2,
                user=db_user2,
                password=db_password2,
                database=db_database2)
                
            # Establish a database connection and create a cursor object
            cursor = conn.cursor()   
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
            status = "Recibido"
            
            '''
            Update the pedidos table
            '''
           
            # Execute an SQL query to update the "status" column of the driver with the specified username to True
            cursor.execute("INSERT INTO pedidos (latitud, longitud, direccion, id_cliente, id_comercio, datetime_pedido, tamaño, status) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",(latitud, longitud, direccion, id_cliente, id_comercio[0], datetime_pedido, tamaño, status))
            # Commit the changes to the database
            conn.commit()
        st.write("Pedido realizado")