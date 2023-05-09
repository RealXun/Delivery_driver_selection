# Import the required modules
import os
import streamlit as st
from streamlit_option_menu import option_menu
from scripts import drivers
from scripts import shops
from scripts import location
from scripts import heatmap
from scripts import orders
import json
import psycopg2
from scripts import dash_menu
from scripts import driver_menu
from scripts import user_menu
from scripts import tienda_menu

st.set_page_config(
    layout="wide",
    initial_sidebar_state="auto"
)
os.environ["STREAMLIT_SERVER_MODE"] = "production"

# Set the path to the project root and the folders for scripts and files
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
src_folder = (PROJECT_ROOT + "/" + "src")

# Get database credentials from environment variables
db_host = os.environ.get('host')
db_user = os.environ.get('user')
db_password = os.environ.get('password')
db_database = os.environ.get('database')

# Establish a connection to the database
conn = psycopg2.connect(
    host=db_host,
    user=db_user,
    password=db_password,
    database=db_database)

# Create a cursor object
cur = conn.cursor()


def call_user(n, p):
    # Execute a query to retrieve the user with the provided username and password
    cur.execute("SELECT tipo FROM usuarios WHERE username = %s AND contraseña = %s", (n, p))

    # Retrieve the first row returned by the query
    row = cur.fetchone()

    # If a row exists, return the value of the tipo column
    if row:
        st.write(row[0]) # This code return the type of the account (admin, user, driver)
        messege = "Login successful"
        return True,messege,row[0]

    # If no row exists, return None
    else:
        messege = "Username not found"
        return False,messege,False

    # Close the cursor and connection
    cur.close()
    conn.close()


# get the current session state of the Streamlit app.
session = st.session_state

# Check if user is not logged in yet
if 'login' not in session:
    # display a text input box for the user to enter their username.
    username = st.text_input('Username')
    # display a password input box for the user to enter their password.
    password = st.text_input('Password', type='password')
    # display a button to submit the form
    submit_button = st.button('Submit')
    
    # Check if submit button is clicked
    if submit_button:
        # Call the "call_user" fun+ction to validate the user's credentials
        status, mess, typ = call_user(username, password)
        
        # If the validation fails, display an error message
        if status == False:
            st.error(mess)
            
        # If the validation succeeds, set the login session state variables and display a success message
        else:
            session['login'] = True
            session['username'] = username
            session['menu'] = typ
            st.success('Login successful')
            # rerun the Streamlit app to update the login state
            st.experimental_rerun()



# If user is already logged in, display the appropriate menu based on their credentials
if 'login' in session:
    # if the user has admin credentials, display the admin menu.
    if session['menu'] == 'admin':
        dash_menu.dash_menu(session['username']) # dash_menu is the admin menu, from there it will call the bodys of the different options

    # if the user has client credentials, display the client menu.
    elif session['menu'] == 'cliente':
        user_menu.user_menu(session['username']) # user_menu is the user menu, from there it will call the bodys of the different options

    #  if the user has driver credentials, display the driver menu.
    elif session['menu'] == 'repartidor':
        driver_menu.driver_menu(session['username']) # driver_menu is the repartidor menu, from there it will call the bodys of the different options 

    #  if the user has driver credentials, display the tienda menu.
    elif session['menu'] == 'comercio':
        # Execute the query to fetch the shop id
        cur.execute("SELECT id_comercio FROM usuarios WHERE username = %s", (session['username'],))
        id_ = cur.fetchall()
        id_comercios = id_[0][0]
        tienda_menu.tienda_menu(session['username'],id_comercios) # driver_menu is the tienda menu, from there it will call the bodys of the different options 
