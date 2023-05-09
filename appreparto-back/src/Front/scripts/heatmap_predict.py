import streamlit as st
import os
from pathlib import Path
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium, folium_static
import json
from streamlit_option_menu import option_menu
from scripts import formulas
import psycopg2
import pandas as pd
from folium.features import GeoJsonTooltip
import geopandas as gpd

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


def on_choro_click(feature, **kwargs):
    # Handle the click event here
    st.write(f"Clicked on {feature['properties']['nombre']}")

def heatmap_predict(name):
    #Add the cover image for the cover page. Used a little trick to center the image
             # To display the header text using css style

    st.markdown(""" <style> .font {
        font-size:40px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.write("Mapa de calor de la predicción de demanda por zonas")
    
    # Establish a database connection and create a cursor object
    cursor = conn.cursor()

    # Load the driver data where status is "Entregado"
    cursor.execute("SELECT * FROM pedidos_modelo WHERE status = %s ", ("Entregado",))

    # Fetch all the results from the cursor
    result = cursor.fetchall()

    # Create a pandas dataframe from the query result with column names obtained from cursor description
    df = pd.DataFrame(result, columns=[desc[0] for desc in cursor.description])

    # Load the GeoJSON file into a dictionary
    with open('barrios2.geojson') as f:
        geojson_data = json.load(f)

    # Create a map centered on the geographic coordinates of Madrid
    madrid_map = folium.Map(location=[40.4168, -3.7038], zoom_start=12, tiles=None, overlay=False)

    # Create two FeatureGroup layers for Heatmap and Hot points
    fg0 = folium.FeatureGroup(name='Heatmap',overlay=False).add_to(madrid_map)
    fg1 = folium.FeatureGroup(name='Hot points',overlay=False).add_to(madrid_map)

    # Create two more FeatureGroup layers for Choropleth maps
    fg2 = folium.FeatureGroup(name='Choropleth map',overlay=False).add_to(madrid_map)
    fg3 = folium.FeatureGroup(name='Choropleth map 2',overlay=False).add_to(madrid_map)

    
    '''
    Figura 0
    The code creates a heatmap layer on a Folium map using data obtained from an API call 
    to get driver information. The code first creates a pandas dataframe using the API call 
    data, then merges it with an existing dataframe based on a common column. The code then 
    extracts latitude and longitude coordinates from the merged dataframe and creates a list 
    of tuples. Finally, the HeatMap function is used to create a heatmap layer on a Folium map 
    using the list of coordinates, which is then added to the map object.
    '''
    # Create a pandas dataframe using data obtained from an API call to get driver information
    df_comercios = pd.DataFrame(formulas.get_comercios_api())
    
    # Merge the two DataFrames based on the 'Id' column
    df_merged = pd.merge(df, df_comercios, left_on='id_comercio', right_on='id', how='left')
    
    # Extract coordinates as a list of tuples
    coordinates = list(zip(df_merged["latitud"], df_merged["longitud"]))

    # Add a heatmap layer to the map using the coordinates list
    HeatMap(coordinates).add_to(fg0)

  
    '''
    Figura 1
    The code reads in a CSV file into a pandas dataframe, groups the data by neighborhood and calculates 
    the mean of the latitud and longitud columns for each neighborhood. Then, it creates a heatmap layer 
    using these coordinates on a folium map object.
    '''
    # Importing a CSV file named "pedidos_modelo_with_zona2.csv" into a pandas dataframe named "pedidos_modelo_with_zona"
    pedidos_modelo_with_zona = pd.read_csv("pedidos_modelo_with_zona2.csv") 

    # Grouping the dataframe by neighborhood and calculating the mean of the latitud and longitud columns, then resetting the index and selecting only the neighborhood, latitud, and longitud columns.
    df_grouped = pedidos_modelo_with_zona.groupby('neighborhood').mean().reset_index()[['neighborhood', 'latitud', 'longitud']]

    # Creating a heatmap layer on top of a folium map object named "fg1" using the latitud and longitud columns of the grouped dataframe, with a radius of 15.
    HeatMap(data=df_grouped[['latitud', 'longitud']], radius=15).add_to(fg1)
    
    
    '''
    Figura 2
    This code loads data from a CSV file into a pandas dataframe, groups the data by neighborhood, 
    and counts the number of records in each group. It then creates a Choropleth map layer using 
    the neighborhood count data, a GeoJSON file, and specified map styling options, and adds it 
    to a folium map object named 'fg2'. The resulting map visualizes the number of records per neighborhood.
    '''
    # Load the data from a CSV file into a pandas dataframe
    data = pd.read_csv('pedidos_modelo_with_zona2.csv')

    # Group the data by neighborhood and count the number of records in each group, then reset the index and rename the count column to 'count_by_zona'
    count_by_zona = data.groupby('neighborhood').size().reset_index(name='count')

    # Create a Choropleth map layer using the geojson_data, neighborhood count data, and specified map styling options, and add it to a folium map object named 'fg2'
    choropleth = folium.Choropleth(
            geo_data=geojson_data, # Path or URL to GeoJSON data
            name='choropleth', # Name of the layer
            data=count_by_zona, # Data to visualize on the map
            columns=['neighborhood', 'count'], # Column names to use for the keys in the 'data' argument
            key_on='feature.properties.nombre', # Key in the 'geo_data' dictionary that is used to join with the data
            fill_color='YlGn', # Color scale to use for the fill color of polygons
            fill_opacity=0.7, # Opacity of the fill color
            line_opacity=0.2, # Opacity of the borders of polygons
            legend_name='Number of records', # Title of the legend
            highlight=True, # Whether to highlight the polygons on mouseover
            smooth_factor=0.1 # The degree of smoothing to apply to polygon edges
        ).geojson.add_to(fg2) # Add the layer to the map object named 'fg2'
    
   
    '''
    Figura 3
    The code loads a GeoJSON file and creates a choropleth map using the count data from a CSV file. 
    The script filters the GeoJSON data based on the neighborhood names, and creates a GeoDataFrame. 
    It then merges the count data with the GeoDataFrame and creates a choropleth map using Folium. 
    The resulting map shows the number of orders per neighborhood.
    '''
    # Load the GeoJSON file into a dictionary
    with open('barrios2.geojson') as f:
        geojson_data = json.load(f)

    # Create a Pandas DataFrame with the count data
    count_by_zona = pd.read_csv('pedidos_modelo_with_zona2.csv')

    # Group the data by neighborhood and count the number of occurrences
    count_by_zona = count_by_zona.groupby('neighborhood').size().reset_index(name='count')

    # Replace "AZCA" with "Azca" in the 'zona' column
    count_by_zona['neighborhood'] = count_by_zona['neighborhood'].replace('AZCA', 'Azca')

    # Rename the 'neighborhood' column to 'nombre'
    count_by_zona = count_by_zona.rename(columns={'neighborhood': 'nombre'})

    # Group the count data by neighborhood and sum the counts
    data = count_by_zona.groupby('nombre')['count'].sum().reset_index()

    # Get a list of neighborhood names from the count_by_zona DataFrame
    neighborhood_names = count_by_zona['nombre'].tolist()

    # Filter the GeoJSON features to remove any that don't match the neighborhood names
    filtered_features = []
    for feature in geojson_data['features']:
        if feature['properties']['nombre'] in neighborhood_names:
            filtered_features.append(feature)
    geojson_data['features'] = filtered_features

    # Check if there are any neighborhood names in count_by_zona that are not in geojson_data
    geojson_names = [feature['properties']['nombre'] for feature in geojson_data['features']]
    missing_names = set(neighborhood_names) - set(geojson_names)

    # Create a GeoDataFrame from the filtered GeoJSON data
    gdf = gpd.GeoDataFrame.from_features(filtered_features)

    # Merge the count data with the GeoDataFrame based on the neighborhood name
    count_by_zona = count_by_zona.merge(gdf[['nombre', 'geometry']], on='nombre')

    # Create a choropleth map using Folium
    choro = folium.Choropleth(                # Create a choropleth map using the folium library
            geo_data=geojson_data,            # The GeoJSON data for the map
            name='choropleth',                # Name of the choropleth map
            data=data,                        # Data to be plotted on the map
            columns=['nombre', 'count'],      # Columns to be used for plotting the data
            key_on='feature.properties.nombre', # The key for matching the data with the GeoJSON properties
            fill_color='YlGn',                # Color for the map based on data values
            fill_opacity=0.7,                 # Opacity of the map fill color
            line_opacity=0.2,                 # Opacity of the boundary lines of the map
            legend_name='Count',              # Name of the legend for the map
            highlight=True,                   # Whether to highlight the selected feature on the map
            reset=True,                       # Whether to reset the map when a new feature is clicked
            smooth_factor=0.1,                # The degree of smoothing to apply to polygon edges
            on_each_feature=on_choro_click,   # A function to be called for each feature on the map
        ).geojson.add_to(fg3)                 # Add the choropleth map to a folium FeatureGroup


    # Set the name of the popup fields
    name = ' '.join(['Distrito:', 'Barrio:', 'N Pedidos:'])


    '''
    Create a HeatMap layer using the latitud and longitud columns of the previously created 
    grouped dataframe and add it to a folium map object named 'madrid_map', along with a couple of TileLayer objects and a LayerControl object
    '''
    # This line creates a GeoJson layer from a GeoJSON file named 'barrios2.geojson' and adds it to a Choropleth map object named 'choro', with a specified name and a popup window that shows properties from the GeoJSON features and a column from the count_by_zona dataframe.
    folium.features.GeoJson('barrios2.geojson',name=name,popup=folium.features.GeoJsonPopup(fields=['nomdis', 'nombre','count'])).add_to(choro)
    
    # Create the HeatMap layer and add it to the map object
    HeatMap(data=df_grouped[['latitud', 'longitud']], radius=15).add_to(madrid_map)
    
    # Add a TileLayer with a dark theme to the map object
    folium.TileLayer('cartodbdark_matter',overlay=True,name="View in Dark Mode").add_to(madrid_map)
    
    # Add a TileLayer with a light theme to the map object
    folium.TileLayer('cartodbpositron',overlay=True,name="View in Light Mode").add_to(madrid_map)
    
    # Add a LayerControl object to the map object, which allows the user to toggle the display of different map layers 
    folium.LayerControl().add_to(madrid_map) 
    
    
    
    
    # Display the map
    folium_static(madrid_map,width=1400, height=1000)