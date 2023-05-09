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

# Get database credentials from environment variables
db_host2 = os.environ.get('hostnew')
db_user2 = os.environ.get('user')
db_password2 = os.environ.get('password')
db_database2 = os.environ.get('database')


def pedido_reparto(name):
    #Add the cover image for the cover page. Used a little trick to center the image
             # To display the header text using css style

    st.markdown(""" <style> .font {
        font-size:40px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.write("Pedidos en reparto") 

    # Create a Pandas DataFrame from the data retrieved from the API
    df_pedidos_en_reparto = pd.DataFrame(formulas.get_pedidos_api())
    
    # Filter the DataFrame to include only orders with status "Recibido"
    df_pedidos_en_reparto = df_pedidos_en_reparto[df_pedidos_en_reparto['status'] == "En reparto"]
    
    # Rename the selected columns to "ID Pedido", "Estado", and "Fecha Pedido"
    df_pedidos_en_reparto = df_pedidos_en_reparto.rename(columns={'id_pedido': 'ID Pedido', 'id_cliente':'ID Cliente','id_comercio': 'ID Comercio', 'id_repartidor':'ID Repartidor','datetime_pedido': 'Fecha','status': 'Estado','tamaño':'Tamaño','direccion':'Direccion'})
    
    # Select only the columns "id_pedido", "status", and "datetime_pedido"
    df_pedidos_en_reparto = df_pedidos_en_reparto[['ID Pedido', 'ID Cliente', 'ID Comercio', 'ID Repartidor','Fecha', 'Estado', 'Tamaño', 'Direccion']]
    
    # Sort the DataFrame by "ID Pedido" in ascending order
    df_pedidos_en_reparto = df_pedidos_en_reparto.sort_values(by=['ID Pedido'], ascending=True)
    
    # Reset the DataFrame index to start from 0
    df_pedidos_en_reparto = df_pedidos_en_reparto.reset_index(drop=True)
    
    # Display the DataFrame in Streamlit using the set_index() method to set the index to "ID Pedido"# Establish a database connection and create a cursor object
    st.dataframe(df_pedidos_en_reparto.set_index('ID Pedido'))

    
    # Prompts the user to enter an oder id 
    # get and stores the input in the 'user_input' variable.    
    user_input = st.text_input("Que id de pedido quiere modificar?")
    
    # The input is then converted into an integer type and stored in the 
    # 'id_chosen' variable using the 'int' function. 
    try:
        id_chosen = int(user_input)
    except ValueError:
        st.error('Completa el campo con un numero entero para poder realizar la acción')
        
        
    option = ['Recibido','Entregado']
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
        # Establish a connection to the database
            conn = psycopg2.connect(
                host=db_host2,
                user=db_user2,
                password=db_password2,
                database=db_database2)

            # Establish a database connection and create a cursor object
            cursor = conn.cursor()  
            # Filter the DataFrame to keep only the row(s) where id_pedido matches the given value
            filtered_df = df_pedidos_en_reparto[df_pedidos_en_reparto["ID Pedido"] == int(user_input)]
            # Extract the value(s) of id_repartidor from the filtered DataFrame
            id_repartidor_values = filtered_df["ID Repartidor"].values
            

            # Check if id_repartidor_values is not empty and does not contain NaN values
            if id_repartidor_values is not None and not np.isnan(id_repartidor_values[0]):
                # Convert id_repartidor_values to an integer (assuming there's only one value)
                id_repartidor = int(id_repartidor_values[0])
                # Update the "ocupado" value in the database for the specified email
                cursor.execute("UPDATE repartidores SET ocupado = %s WHERE id = %s", (False, id_repartidor))
                # Update the "status" value in the "pedidos" table to False for the driver with the specified username
                cursor.execute("UPDATE pedidos SET status = %s WHERE id_pedido = %s", (selected_option, id_chosen))
            else:
                # Update the "status" value in the "pedidos" table to False for the driver with the specified username
                cursor.execute("UPDATE pedidos SET status = %s WHERE id_pedido = %s", (selected_option, id_chosen))


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