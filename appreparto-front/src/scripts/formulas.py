import json
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
from geopy.geocoders import Nominatim
import math
import datetime
import geopandas as gpd
import numpy as np


PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
scripts_folder = (PROJECT_ROOT + "/" + "scripts")
data_folder = (PROJECT_ROOT + "/" + "data")
raw_data = (data_folder + "/" + "raw_data")
processed_data = (data_folder + "/" + "processed_data")

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

'''
This code defines a function get_pedidos_api() that retrieves data from an API endpoint 
containing information about orders. The data is then processed and stored in a list of 
dictionaries. Each dictionary contains the relevant fields for an order, such as ID, client, 
commerce, location, timestamp, size, status, and delivery address. The function returns 
the list of dictionaries containing the processed order information.
'''
def get_pedidos_api():
    # Retrieve data from API endpoint and convert it to JSON
    data = requests.get("https://eogwk5ap22.execute-api.eu-north-1.amazonaws.com/Dev/pedidos").json()
    # Create an empty list to store the dictionaries of pedidos
    pedidos_list = []
    # Iterate over each pedido in the data and extract the relevant information into a dictionary
    for char in data:
        dict_pedidos = {
            'id_pedido': char["id_pedido"] if char["id_pedido"] else None, # If id_pedido is present in the char dictionary, store its value. Otherwise, store None.
            'id_cliente': char["id_cliente"] if char["id_cliente"] else None, # If id_cliente is present in the char dictionary, store its value. Otherwise, store None.
            'id_comercio': char["id_comercio"] if char["id_comercio"] else None, # If id_comercio is present in the char dictionary, store its value. Otherwise, store None.
            'id_repartidor': char["id_repartidor"] if char["id_repartidor"] else None, # If id_repartidor is present in the char dictionary, store its value. Otherwise, store None.
            'latitud': char["latitud"] if char["latitud"] else None, # If latitud is present in the char dictionary, store its value. Otherwise, store None.
            'longitud': char["longitud"] if char["longitud"] else None, # If longitud is present in the char dictionary, store its value. Otherwise, store None.
            'datetime_pedido': char["datetime_pedido"] if char["datetime_pedido"] else None, # If datetime_pedido is present in the char dictionary, store its value. Otherwise, store None.
            'tamaño': char["tamaño"] if char["tamaño"] else None, # If tamaño is present in the char dictionary, store its value. Otherwise, store None.
            'status': char["status"]  if char["status"] else None, # If status is present in the char dictionary, store its value. Otherwise, store None.
            'direccion': char["direccion"]  if char["direccion"] else None # If direccion is present in the char dictionary, store its value. Otherwise, store None.
        }
        # Add the dictionary of pedidos to the list of pedidos
        pedidos_list.append(dict_pedidos)
    # Return the list of pedidos dictionaries
    return pedidos_list




def get_clientes_api():
    # Retrieve data from API endpoint and convert it to JSON
    data = requests.get("https://eogwk5ap22.execute-api.eu-north-1.amazonaws.com/Dev/clientes").json()
    # Create an empty list to store the dictionaries of pedidos
    clientes_list = []
    # Iterate over each pedido in the data and extract the relevant information into a dictionary
    for char in data:
        dict_clientes = {
            'id': char["id"] if char["id"] else None, # If id is present in the char dictionary, store its value. Otherwise, store None.
            'nombre': char["nombre"] if char["nombre"] else None, # If nombre is present in the char dictionary, store its value. Otherwise, store None.
            'email': char["email"] if char["email"] else None, # If email is present in the char dictionary, store its value. Otherwise, store None.
            'direccion': char["direccion"] if char["direccion"] else None, # If direccion is present in the char dictionary, store its value. Otherwise, store None.
            'latitud': char["latitud"] if char["latitud"] else None, # If latitud is present in the char dictionary, store its value. Otherwise, store None.
            'longitud': char["longitud"] if char["longitud"] else None # If longitud is present in the char dictionary, store its value. Otherwise, store None.
        }
        # Add the dictionary of pedidos to the list of pedidos
        clientes_list.append(dict_clientes)
    # Return the list of pedidos dictionaries
    return clientes_list




'''
This code defines a function get_repartidores_api() that sends a GET request to an API endpoint 
to retrieve data on delivery drivers. It then processes the JSON response and extracts specific 
fields such as the driver's ID, name, email, status, vehicle type, location, and last update time. 
It stores this information for each driver in a dictionary, which is appended to a list of drivers. 
Finally, it returns the list of drivers as the output of the function.
'''    
def get_repartidores_api():
    # Get the data from the specified URL and save it in JSON format
    data = requests.get("https://eogwk5ap22.execute-api.eu-north-1.amazonaws.com/Dev/repartidores").json()
    # Create an empty list to hold the driver details
    repartidores_list = []
    # Loop through each driver in the retrieved data and extract the relevant information
    for char in data:
        dict_repartidores = {'id': char["id"],
                    'nombre': char["unombre"] if char["unombre"] else None,
                    'email': char["email"] if char["email"] else None,
                    'status': char["status"] if char["status"] else None,
                    'vehiculo': char["vehiculo"] if char["vehiculo"] else None,
                    'ocupado': char["ocupado"] if char["ocupado"] else None,
                    'zona': char["zona"] if char["zona"] else None,
                    'latitud': char["latitud"] if char["latitud"] else None,
                    'longitud': char["longitud"] if char["longitud"] else None,
                    'datetime_ult_act': char["datetime_ult_act"]  if char["datetime_ult_act"] else None
                    }
        # Add the extracted driver details to the list
        repartidores_list.append(dict_repartidores)
    # Return the list of driver details
    return repartidores_list   




'''
This function calls an API to retrieve data about commercial establishments and transform it into 
a list of dictionaries. The dictionaries contain information about the establishment's 
ID, name, area, address, type, latitude, and longitude.
'''
def get_comercios_api():
    # Use the requests library to send a GET request to the API endpoint and receive JSON data in response
    data = requests.get("https://eogwk5ap22.execute-api.eu-north-1.amazonaws.com/Dev/comercios").json()

    # Create an empty list to store the formatted data
    comer_list = []

    # Loop through each dictionary in the received data and format it into a new dictionary with desired keys
    for char in data:
        dict_comer = {'id': char["id"],
                    'nombre': char["nombre"],
                    'zona': char["zona"],
                    'direccion': char["direccion"],
                    'tipo': char["tipo"],
                    'latitud': char["latitud"],
                    'longitud': char["longitud"]
                    }

        # Append the new dictionary to the list
        comer_list.append(dict_comer)

    # Return the list of formatted dictionaries
    return comer_list


def get_time_distance(origen_long, origen_lat, destin_long, destin_lat,*lugar):
    # Build the API request URL with the origin, destination, and API key
    url = "https://api.openrouteservice.org/v2/directions/driving-car?api_key=5b3ce3597851110001cf6248037ed40de1b6444ca5b189715abad70e&start={},{},&end={},{}".format(origen_long, origen_lat, destin_long, destin_lat)
    
    # Make the API request and parse the JSON response
    response = requests.get(url)
    data = json.loads(response.text)
    
    # Extract the distance and duration from the API response
    distance = data['features'][0]['properties']['segments'][0]['distance'] / 1000  # convert to kilometers
    duration = data['features'][0]['properties']['segments'][0]['duration'] / 60  # convert to minutes
    if lugar[0] == "tienda":
        messege = 'La distancia al lugar de recogida es de : {:.2f} kilometros'.format(distance),'La duración estimada del trayecto es de: {:.2f} minutos'.format(duration)
    else :
        messege = 'La distancia al lugar de entrega desde el punto de recogida es de : {:.2f} kilometros'.format(distance),'La duración estimada del trayecto es de: {:.2f} minutos'.format(duration)
    
    return messege
  


def name_unique(df,name):
    '''
    Return a list of unique names of type of shops
    '''
    if name == "tipos":
        df = df[df["tipos"] != "restaurant"] # ESTO HAY QUE ELIMINARLO UNA VEZ MODIFICADA LA BASE DE DATOS DE COMERCIOS
        df = df["tipos"].unique().tolist()
    else:
        df = df["nombre"].unique().tolist()   
    return df
    
    
 

def generador_direccion():
    """
    Función que genera una dirección aleatoriamente en un radio de 10km centrado
    en las coords del centro de Madrid. Para ello calcula un punto con la fórmula
    de distancias de lat y long gracias a la librería math y usa la librería 
    geocoders para obtener direcciones y recalcular coordenadas en base a esta.
    """

    # Create an instance of Nominatim class for geocoding/reverse-geocoding
    geolocator = Nominatim(user_agent=db_user)

    # Set the coordinates of the center point (Madrid) in radians
    centro = (40.4168, -3.7038)
    lat_madrid = math.radians(centro[0])
    lon_madrid = math.radians(centro[1])

    # Set a flag to control the generation of a valid address
    direccion_flag = True
    while direccion_flag:
        # Generate a random distance and angle from the center point
        d = np.random.uniform(0, 10)
        ang = np.random.uniform(0, 2*np.pi)
        R = 6371
        # Calculate the new latitude and longitude based on the random distance and angle
        lat = math.asin(math.sin(lat_madrid) * math.cos(d/R) + 
                        math.cos(lat_madrid) * math.sin(d/R) * math.cos(ang))
        lon = lon_madrid + \
            math.atan2(math.sin(ang) * math.sin(d/R) * math.cos(lat_madrid), 
                       math.cos(d/R) - math.sin(lat_madrid) * math.sin(lat))
        # Convert the new latitude and longitude to degrees
        lat = math.degrees(lat)
        lon = math.degrees(lon)

        # Use the geolocator to reverse-geocode the latitude and longitude into an address
        direccion = geolocator.reverse((lat, lon))
        try:
            # Check if the first element of the address is a valid street number
            int(direccion[0].split(",")[0])
            try:
                # Check if the second element of the address is a valid street name
                int(direccion[0].split(",")[1])
            except:
                # If the second element is not a valid street name, use the latitude and longitude as backup
                latitud = direccion[1][0]
                longitud = direccion[1][1]
                print(latitud)
                print(longitud)
                # Format the address as "<street name>, <street number>, Madrid"
                direccion = "{}, {}, Madrid" \
                    .format(direccion[0].split(",")[1].strip(), 
                        direccion[0].split(",")[0])
                direccion_flag = False
        except:
            # If the first element is not a valid street number, continue the loop
            pass

    # Return the valid address, latitude, and longitude
    return direccion, latitud, longitud
    
    
'''
This function finds the nearest available delivery driver to a given shop location using the k-Nearest Neighbors algorithm. 
It takes the ID of a shop as input and returns the ID of the closest available driver. The function retrieves the latitude 
and longitude of the shop and delivery drivers from API calls, calculates the Manhattan distance between each driver and the 
shop, fits a k-NN model to the driver locations, and returns the ID of the closest available driver to the shop location.
'''
def find_driver(id_comer):

    # Get the comercios data and filter by id_comer
    df_comercios = pd.DataFrame(get_comercios_api())
    df_comercios = df_comercios[df_comercios['id'] == id_comer]

    # Get the repartidores data and filter by status=True and ocupado!=True
    df_repartidores = pd.DataFrame(get_repartidores_api())
    df_repartidores = df_repartidores[(df_repartidores['status'] == True) & (df_repartidores['ocupado'] != True)]

    # Get the latitude and longitude of the shop from the first row of the df_comercios DataFrame
    shop_lat = df_comercios["latitud"].values[0]
    shop_lon = df_comercios["longitud"].values[0]

    # Calculate the Manhattan distance between each delivery driver and the shop using their latitudes and longitudes
    df_repartidores["manhattan_distance"] = np.abs(df_repartidores["latitud"] - shop_lat) + np.abs(df_repartidores["longitud"] - shop_lon)

    # Set the number of nearest neighbors to consider for the k-NN algorithm
    k = 1

    # Create a matrix X containing the latitude and longitude of each delivery driver
    matrix = df_repartidores[["latitud", "longitud"]]

    # add column names to matrix
    matrix.columns = ["latitud", "longitud"]

    # Reset the index of the df_repartidores DataFrame
    df_repartidores = df_repartidores.reset_index(drop=True)

    # Fit the k-NN model to the matrix
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='manhattan').fit(matrix)

    # Find the indices and distances of the k nearest neighbors to the shop location
    shop_location = [[shop_lat, shop_lon]]
    shop_location_df = pd.DataFrame(shop_location, columns=['latitud', 'longitud'])  # add column names to the shop location

    # Find the indices and distances of the k nearest neighbors to the shop location
    distances, indices = nbrs.kneighbors(shop_location_df)

    # Get the index of the closest driver from the indices array
    closest_driver_index = indices[0][0]

    # Get the ID of the closest driver from the df_repartidores DataFrame using the closest driver's index
    closest_driver_id = df_repartidores.loc[closest_driver_index, "id"]

    # Return the ID of the closest driver 
    return closest_driver_id
    
    
'''
This code connects to a PostgreSQL database and fetches data from two tables. It also reads 
a CSV file and performs data wrangling tasks like merging columns, dropping columns, and 
selecting specific columns. Finally, it merges two dataframes on specific columns and saves 
the resulting dataframe as a CSV file.
'''    
    
def data_for_prediccion(franja_horaria):
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
    '''
    This code connects to a database using a cursor object, then executes a SQL query to 
    fetch all the rows from a table named pedidos_modelo where the status column is equal 
    to "Entregado". It then creates a Pandas dataframe using the results returned by the query
    , where the column names are obtained from the cursor description. Finally, it saves the 
    dataframe as a CSV file in a specified directory.
    '''
    ### Get pedidos_modelo
    # Establish a database connection and create a cursor object
    cursor = conn.cursor()
    # Load the driver data where status is "Entregado"
    cursor.execute("SELECT * FROM pedidos_prediccion")
    # Fetch all the results from the cursor
    result = cursor.fetchall()
    # Create a pandas dataframe from the query result with column names obtained from cursor description
    df_pedidos_modelo = pd.DataFrame(result, columns=[desc[0] for desc in cursor.description])

    # save df
    df_pedidos_modelo.to_csv(raw_data + '/' + 'df_pedidos_modelo.csv', index=False)


    '''
    This code connects to a database using a cursor object, then executes a SQL query to 
    fetch all the rows from a table named pedidos_prediccion where the status column is 
    equal to "Entregado". It then creates a Pandas dataframe using the results returned by the 
    query, where the column names are obtained from the cursor description. Finally, it saves 
    the dataframe as a CSV file in a specified directory.
    '''
    # Establish a database connection and create a cursor object
    cursor = conn.cursor()

    # Load the driver data where status is "Entregado"
    cursor.execute("SELECT * FROM pedidos_prediccion")

    # Fetch all the results from the cursor
    result = cursor.fetchall()

    # Create a pandas dataframe from the query result with column names obtained from cursor description
    df_prediccion = pd.DataFrame(result, columns=[desc[0] for desc in cursor.description])

    # save df
    df_prediccion.to_csv(raw_data + '/' + 'df_prediccion.csv', index=False)

    '''
    The code reads a CSV file containing the current addresses of Madrid with geographic coordinates, 
    merges two columns to create a new column representing the district and neighborhood codes, 
    and drops the original columns. The resulting DataFrame contains the district and neighborhood 
    names, the section census code, and the merged district and neighborhood codes.
    '''
    # Opening Direcciones postales vigentes con coordenadas geográficas file from https://datos.madrid.es/ (already downloaded)
    # Specify the columns to keep
    columns_to_keep = ['Nombre del distrito','Nombre del barrio','Codigo de distrito', 'Codigo de barrio', 'Seccion censal','Codigo postal']

    # Read the CSV file into a Pandas dataframe, using only the specified columns
    # Set the delimiter to ';' and the encoding to ISO-8859-1
    df_madrid_es = pd.read_csv(raw_data + '/' + 'CALLEJERO_VIGENTE_NUMERACIONES_202303.csv', usecols=columns_to_keep, sep=';', encoding='ISO-8859-1')

    # Merge columns 'Codigo de distrito' and 'Codigo de barrio' into a new column 'codbarrio'
    df_madrid_es['codbarrio'] = df_madrid_es.apply(lambda x: str(x['Codigo de distrito']) + '-' + str(x['Codigo de barrio']), axis=1)

    # Drop the columns 'Codigo de distrito' and 'Codigo de barrio' from the dataframe
    df_madrid_es = df_madrid_es.drop(['Codigo de distrito', 'Codigo de barrio'], axis=1)

    # save df
    df_madrid_es.to_csv(processed_data + '/' + 'df_madrid_es.csv', index=False)


    '''
    This code reads in two CSV files: 'df_madrid_es.csv' and 'df_prediccion.csv', then it merges 
    them on columns 'Codigo postal' and 'zona' and renames the 'codbarrio' column to 
    'cod_barrio'. After that, it selects the necessary columns and saves the resulting dataframe 
    to a new CSV file named 'df_prediccion_with_codbarrio.csv'. 
    '''
    # Read df_madrid_es
    df_madrid_es = pd.read_csv(processed_data + '/' + 'df_madrid_es.csv')
    # Opening df_madrid_es data
    df_prediccion = pd.read_csv(raw_data + '/' + 'df_prediccion.csv')

    # Merge df_madrid_es with df_prediccion on 'Codigo postal' and 'zona'
    df_merged = pd.merge(df_madrid_es, df_prediccion, left_on='Codigo postal', right_on='zona')

    # Rename the 'codbarrio' column to 'cod_barrio'
    df_merged.rename(columns={'codbarrio': 'cod_barrio'}, inplace=True)

    # Select only the necessary columns
    df_final = df_merged[['Nombre del distrito', 'Nombre del barrio', 'Seccion censal', 'cod_barrio','zona','franja_horaria','total','fecha']]

    # Save the resulting dataframe to a new CSV file
    df_final.to_csv(processed_data + '/' + 'df_prediccion_with_codbarrio.csv', index=False)

    '''
    The code reads a CSV file containing data on predicted air quality levels in different 
    neighborhoods of Madrid, filters the data for a specific date and time range, and counts the 
    number of data points for each neighborhood. The resulting dataframe is then saved to a 
    new CSV file.
    '''
    # get today's date in Spanish time
    today = datetime.datetime.now().strftime('%Y-%m-%d')

    # Read df_prediccion_with_codbarrio
    df_prediccion_with_codbarrio = pd.read_csv(processed_data + '/' + 'df_prediccion_with_codbarrio.csv')

    # Select only rows with franja_horaria 6 and date 2022-03-31
    df_prediccion_with_codbarrio = df_prediccion_with_codbarrio[(df_prediccion_with_codbarrio["franja_horaria"] == franja_horaria) & (df_prediccion_with_codbarrio["fecha"] == today)]

    # Group the rows by cod_barrio and count the occurrences
    count_by = df_prediccion_with_codbarrio.groupby('cod_barrio')['total'].sum().reset_index(name='count')

    # Save the resulting dataframe to a new CSV file
    count_by.to_csv(processed_data + '/' + 'df_prediccion_with_count.csv', index=False)

    '''
    This code reads a CSV file with counts of some variable by neighborhood, and a GeoJSON 
    file with information about each neighborhood. Then, it iterates through each neighborhood 
    in the GeoJSON file and if the neighborhood code matches a code in the CSV file, it sets a 
    property with the count for that neighborhood. After that, it filters the GeoJSON features to 
    remove any neighborhoods that are not in the CSV file, and saves the resulting GeoJSON 
    file. This process adds a count property to each neighborhood in the GeoJSON file based on 
    the counts in the CSV file.
    '''
    # load count_by dataframe from CSV file
    count_by = pd.read_csv(processed_data + '/' + 'df_prediccion_with_count.csv')

    # load barrios.geojson as a GeoDataFrame
    barrios = gpd.read_file(raw_data + '/' + 'barrios.geojson')

    # iterate through each feature in barrios GeoDataFrame
    for index, row in barrios.iterrows():
        # get the codbarrio property value for the current feature
        cod_barrio = row['codbarrio']
        # find the corresponding row in count_by dataframe
        count_row = count_by[count_by['cod_barrio'] == cod_barrio]
        # if the row exists, set the count property for the feature
        if not count_row.empty:
            barrios.loc[index, 'count'] = int(count_row['count'].iloc[0])

    # remove features with count = 0
    barrios_with_counts = barrios[barrios['count'] != 0]
    # remove rows with non-finite count values
    barrios_with_counts = barrios_with_counts[np.isfinite(barrios_with_counts['count'])]
    # convert count column to integer type
    barrios_with_counts['count'] = barrios_with_counts['count'].astype(int)

    # save the result as a new geojson file
    barrios_with_counts.to_file(processed_data + '/' + 'barrios_with_counts2.geojson', driver='GeoJSON')
    # Close the cursor