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
import math

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

'''
Main body of the page
'''
def drivers_info(name):
    # Add CSS style to change the font size, family, and color of the text displayed by the function
    st.markdown(""" <style> .font {
        font-size:40px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)

    # Retrieve the status of the driver with the specified username(the username for login is the email)
    username = name 
    '''
    This part of the code are the top messeges
    '''

    # Define the column layout using beta_columns()    
    col1, col2 = st.columns(2) # Create two columns
    with col1:
        '''
        This code retrieves the status of a driver with a specified username (the username for login is the email) from a database table, 
        and displays whether the driver is currently working or not working based on the retrieved status. 
        If the driver is not found in the database, a message is displayed indicating that the user is not found. 
        '''


        # Get the data from the API and create a DataFrame
        df_repartidores = pd.DataFrame(formulas.get_repartidores_api())

        # Filter the DataFrame to keep only the row(s) where email matches the given value
        filtered_df = df_repartidores[df_repartidores["email"] == username]

        # Extract the value(s) of "status" from the filtered DataFrame
        status_values = filtered_df["status"].values

        # Check if status_values is not empty
        if status_values is not None:

            status = status_values[0]

            # If False, print a message
            if status == False:
                st.warning("No estás trabajando")

            elif status == True:
                st.markdown('<h1 style="color: green;">Estás en horario laboral</h1>', unsafe_allow_html=True)  
                
            # If True, print a message
            else:
                st.warning("No estás trabajando")
          
        else:
            # If the specified username is not found in the database, display an error message
            st.write("Usuario no encontrado")

        
        
        
    with col2:       
        '''
        This code retrieves the "ocupado" value from the database for a given driver, which indicates 
        whether they are currently delivering or not. If the value is found, it displays a message 
        indicating whether they are currently delivering or not. If the value is not found, it displays 
        a message indicating that the user was not found.
        '''
        
        # Get the data from the API and create a DataFrame
        df_repartidores = pd.DataFrame(formulas.get_repartidores_api())

        # Filter the DataFrame to keep only the row(s) where email matches the given value
        filtered_df = df_repartidores[df_repartidores["email"] == username]

        # Extract the value(s) of "ocupado" from the filtered DataFrame
        status_values = filtered_df["ocupado"].values
        # If the "ocupado" value is found, display whether the driver is currently delivering or not delivering
        if status_values is not None:
            status = status_values[0]
            # If False, print a message
            if status == False:
                st.warning("No estás repartiendo")

            elif status == True:
                st.markdown('<h1 style="color: green;">Estás repartiendo</h1>', unsafe_allow_html=True)  
                
            # If True, print a message
            else:
                st.warning("No estás repartiendo")
          
        else:
            # If the specified username is not found in the database, display an error message
            st.write("Usuario no encontrado")


    '''
    This part of the code are buttons
    '''    
    
    # Define the column layout using beta_columns()    
    col1, col2 = st.columns(2) # Create two columns
    with col1:
        st.write("Cambia tu estado laboral, trabajando o no trabajando")
        '''
        This code block checks if the "No trabajando" button has been clicked. If so, it updates the 
        "status" value in the "repartidores" table to False for the driver with the specified username, 
        and displays a message to indicate that the status has been updated.
        '''
        # If the "No trabajando" button is clicked, update the "status" value to False
        if st.button("No trabajando 😴"):
            # Get the data from the API and create a DataFrame
            df_repartidores = pd.DataFrame(formulas.get_repartidores_api())

            # Filter the DataFrame to keep only the row(s) where email matches the given value
            filtered_df = df_repartidores[df_repartidores["email"] == username]

            # Extract the value(s) of "ocupado" from the filtered DataFrame
            status_values = filtered_df["ocupado"].values

            # If the "ocupado" value is found, display whether the driver is currently delivering or not delivering
            if status_values is not None:
                status = status_values[0]
                # If False, the user is not working, so he cannot update "ocupado" status
                if status == True:
                    st.info("Estas repartiendo, no puedes cambiar a No trabajando.")
                # If True, the user is working, so he can update "ocupado" status    
                else: 
                
                    # Establish a connection to the database
                    conn = psycopg2.connect(
                        host=db_host2,
                        user=db_user2,
                        password=db_password2,
                        database=db_database2)
                    # Establish a database connection and create a cursor object
                    cursor = conn.cursor()
                    # Update the "status" value in the "repartidores" table to False for the driver with the specified username
                    cursor.execute("UPDATE repartidores SET status = %s WHERE email = %s", (False, username))
                    # Commit the changes to the database
                    conn.commit()
                    # Display a message to indicate that the status has been updated
                    st.write("Status updated to No trabajando")
                    # If button is clicked, rerun the Streamlit app
                    st.experimental_rerun()

            
            
        '''
        This code defines a conditional statement that checks whether the "Trabajando" button is clicked. 
        If it is clicked, it updates the "status" value to True in the database for the corresponding user. 
        If the "No trabajando" button is clicked instead, it updates the "status" value to False. It then displays 
        a message to confirm that the status has been updated.
        '''
        # If the "Trabajando" button is clicked, update the "status" value to True
        if st.button("Trabajando 😁"):
            # Establish a connection to the database
            conn = psycopg2.connect(
                host=db_host2,
                user=db_user2,
                password=db_password2,
                database=db_database2)            
            
            # Aquí iria la lógica de cambiar tmb el estado del pedido
            direccion, latitud, longitud = formulas.generador_direccion()
            # Establish a database connection and create a cursor object
            cursor = conn.cursor()     
            # Execute an SQL query to update the "status" column of the driver with the specified username to True
            cursor.execute("UPDATE repartidores SET status = %s,latitud = %s,longitud = %s WHERE email = %s", (True, latitud, longitud, username))
            # Commit the changes to the database
            conn.commit()
            # Close the cursor
            cursor.close()
            # Close the connection
            conn.close()            
            # Display a message indicating that the status has been updated
            st.info("Status updated to trabajando")
            # If button is clicked, rerun the Streamlit app
            st.experimental_rerun()

