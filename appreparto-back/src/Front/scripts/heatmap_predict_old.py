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

    '''
    Codigo donde la tienda mete la informacion del pedido a cambiar
    '''
    # Prompts the user to enter an oder id 
    # get and stores the input in the 'user_input' variable.    
    user_input = st.text_input("¿Que id de pedido quiere modificar?")
    
    # The input is then converted into an integer type and stored in the 
    # 'id_chosen' variable using the 'int' function. 
    try:
        number = int(user_input)
    except ValueError:
        st.error('Completa el campo con un numero entero para poder realizar la acción')    
    
    '''
    Esta parte crea un botón de hacer pedido y actualiza la tabla de pedidos
    '''
    
    criteria_selected = user_input
    
    if st.button('Cambiar la franja horaria', disabled=not criteria_selected):
        with st.spinner('Cambiando la franja......'):
            formulas.data_for_prediccion(number)    
    
    

    # Create a map centered on the geographic coordinates of Madrid
    madrid_map = folium.Map(location=[40.4168, -3.7038], zoom_start=12, tiles=None, overlay=False)

    # Create two FeatureGroup layers for Heatmap and Hot points
    fg0 = folium.FeatureGroup(name='Heatmap',overlay=False).add_to(madrid_map)
    fg1 = folium.FeatureGroup(name='Hot points',overlay=False).add_to(madrid_map)
    
    # Create two more FeatureGroup layers for Choropleth maps
    fg2 = folium.FeatureGroup(name='Choropleth map',overlay=False).add_to(madrid_map)
    #fg3 = folium.FeatureGroup(name='Choropleth map 2',overlay=False).add_to(madrid_map)
    fg4 = folium.FeatureGroup(name='Choropleth map 3',overlay=False).add_to(madrid_map)
    fg5 = folium.FeatureGroup(name='Choropleth map5',overlay=False).add_to(madrid_map)

    
    '''
    Figura 0
    The code creates a heatmap layer on a Folium map using data obtained from an API call 
    to get driver information. The code first creates a pandas dataframe using the API call 
    data, then merges it with an existing dataframe based on a common column. The code then 
    extracts latitude and longitude coordinates from the merged dataframe and creates a list 
    of tuples. Finally, the HeatMap function is used to create a heatmap layer on a Folium map 
    using the list of coordinates, which is then added to the map object.
    '''
    
    # Opening df_pedidos_modelo data
    df = pd.read_csv(processed_data + '/' + 'pedidos_modelo_with_zona2.csv')

    # Load the GeoJSON file into a dictionary
    with open(raw_data + '/' + 'barrios2.geojson') as f:
        geojson_data = json.load(f)
    # Create a pandas dataframe using data obtained from an API call to get driver information
    df_comercios = pd.DataFrame(formulas.get_comercios_api())

    # Merge the two DataFrames based on the 'Id' column
    df_merged = pd.merge(df, df_comercios, left_on='id_comercio', right_on='id', how='left')

    # Extract coordinates as a list of tuples
    coordinates = list(zip(df_merged["latitud_y"], df_merged["longitud_y"]))
    
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
     # This line creates a GeoJson layer from a GeoJSON file named 'barrios2.geojson' and adds it to a Choropleth map object named 'choro', with a specified name and a popup window that shows properties from the GeoJSON features and a column from the count_by_zona dataframe.
    folium.features.GeoJson(geojson_data,name=name,popup=folium.features.GeoJsonPopup(fields=['nomdis', 'nombre'])).add_to(choropleth)   
    #
    #'''
    #Figura 3
    #The code loads a GeoJSON file and creates a choropleth map using the count data from a CSV file. 
    #The script filters the GeoJSON data based on the neighborhood names, and creates a GeoDataFrame. 
    #It then merges the count data with the GeoDataFrame and creates a choropleth map using Folium. 
    #The resulting map shows the number of orders per neighborhood.
    #'''
    ## Load the GeoJSON file into a dictionary
    #with open(processed_data + '/' + 'barrios_with_count.geojson') as f:
    #    geojson_data = json.load(f)
    #    
    ## Read df_prediccion_with_count
    #count_by =  pd.read_csv(processed_data + '/' + 'df_prediccion_with_count.csv')
    #
    ## Create a choropleth map using Folium
    #choro = folium.Choropleth(                # Create a choropleth map using the folium library
    #        geo_data=geojson_data,            # The GeoJSON data for the map
    #        name='choropleth',                # Name of the choropleth map
    #        data=count_by,                        # Data to be plotted on the map
    #        columns=['cod_barrio', 'count'],      # Columns to be used for plotting the data
    #        key_on='feature.properties.codbarrio', # The key for matching the data with the GeoJSON properties
    #        fill_color='YlGn',                # Color for the map based on data values
    #        fill_opacity=0.7,                 # Opacity of the map fill color
    #        line_opacity=0.2,                 # Opacity of the boundary lines of the map
    #        legend_name='Count',              # Name of the legend for the map
    #        highlight=True,                   # Whether to highlight the selected feature on the map
    #        reset=True,                       # Whether to reset the map when a new feature is clicked
    #        smooth_factor=0.1,                # The degree of smoothing to apply to polygon edges
    #        on_each_feature=on_choro_click,   # A function to be called for each feature on the map
    #    ).geojson.add_to(fg3)                 # Add the choropleth map to a folium FeatureGroup
    #
    #
    ## Set the name of the popup fields
    #name = ' '.join(['Distrito:', 'Barrio:'])
    ### This line creates a GeoJson layer from a GeoJSON file named 'barrios2.geojson' and adds it to a Choropleth map object named 'choro', with a specified name and a popup window that shows properties from the GeoJSON features and a column from the count_by_zona dataframe.
    #folium.features.GeoJson('barrios2.geojson',name=name,popup=folium.features.GeoJsonPopup(fields=['nomdis', 'nombre',])).add_to(choro)    
    '''
    Figura 4
    The code loads a GeoJSON file and creates a choropleth map using the count data from a CSV file. 
    The script filters the GeoJSON data based on the neighborhood names, and creates a GeoDataFrame. 
    It then merges the count data with the GeoDataFrame and creates a choropleth map using Folium. 
    The resulting map shows the number of orders per neighborhood.
    '''
    # Load the GeoJSON file into a dictionary
    with open(processed_data + '/' + 'barrios_with_counts2.geojson') as f:
        geojson_data = json.load(f)
        
    # Read df_prediccion_with_count
    count_by =  pd.read_csv(processed_data + '/' + 'df_prediccion_with_count.csv')
    # Define color scale
    # Create a choropleth map using Folium
    choro2 = folium.Choropleth(                # Create a choropleth map using the folium library
            geo_data=geojson_data,            # The GeoJSON data for the map
            name='choropleth',                # Name of the choropleth map
            data=count_by,                        # Data to be plotted on the map
            columns=['cod_barrio', 'count'],      # Columns to be used for plotting the data
            key_on='feature.properties.codbarrio', # The key for matching the data with the GeoJSON properties
            fill_color='YlGnBu',                # Color for the map based on data values
            fill_opacity=0.9,                 # Opacity of the map fill color
            line_opacity=0.2,                 # Opacity of the boundary lines of the map
            legend_name='Count',              # Name of the legend for the map
            highlight=True,                   # Whether to highlight the selected feature on the map
            reset=True,                       # Whether to reset the map when a new feature is clicked
            smooth_factor=0.1,                # The degree of smoothing to apply to polygon edges
            on_each_feature=on_choro_click,   # A function to be called for each feature on the map
        ).geojson.add_to(fg4)                 # Add the choropleth map to a folium FeatureGroup
        # Set the name of the popup fields
    name = ' '.join(['Distrito:', 'Barrio:', 'N Pedidos:'])        
    # This line creates a GeoJson layer from a GeoJSON file named 'barrios2.geojson' and adds it to a Choropleth map object named 'choro', with a specified name and a popup window that shows properties from the GeoJSON features and a column from the count_by_zona dataframe.
    folium.features.GeoJson(geojson_data,name=name,popup=folium.features.GeoJsonPopup(fields=['nomdis', 'nombre','count'])).add_to(choro2)
    


    '''
    Figura 5
    This code loads data from a CSV file into a pandas dataframe, groups the data by neighborhood, 
    and counts the number of records in each group. It then creates a Choropleth map layer using 
    the neighborhood count data, a GeoJSON file, and specified map styling options, and adds it 
    to a folium map object named 'fg2'. The resulting map visualizes the number of records per neighborhood.
    '''
    with open(processed_data + '/' + 'barrios_with_counts2.geojson') as f:
        geojson_data = json.load(f)    
    
    # Load the data from a CSV file into a pandas dataframe
    data = pd.read_csv(processed_data + '/' + 'df_prediccion_with_count.csv')
    
    # Group the data by neighborhood and count the number of records in each group, then reset the index and rename the count column to 'count_by_zona'
    count_by_zona = data.groupby('cod_barrio')['count'].sum().reset_index(name='count')
    
    # Create a Choropleth map layer using the geojson_data, neighborhood count data, and specified map styling options, and add it to a folium map object named 'fg2'
    choropleth_fig5 = folium.Choropleth(
            geo_data=geojson_data, # Path or URL to GeoJSON data
            name='choropleth', # Name of the layer
            data=count_by_zona, # Data to visualize on the map
            columns=['cod_barrio', 'count'], # Column names to use for the keys in the 'data' argument
            key_on='feature.properties.codbarrio', # Key in the 'geo_data' dictionary that is used to join with the data
            fill_color='YlGn', # Color scale to use for the fill color of polygons
            fill_opacity=0.7, # Opacity of the fill color
            line_opacity=0.2, # Opacity of the borders of polygons
            legend_name='Number of records', # Title of the legend
            highlight=True, # Whether to highlight the polygons on mouseover
            smooth_factor=0.5 # The degree of smoothing to apply to polygon edges
        ).geojson.add_to(fg5) # Add the layer to the map object named 'fg2'
     # This line creates a GeoJson layer from a GeoJSON file named 'barrios2.geojson' and adds it to a Choropleth map object named 'choro', with a specified name and a popup window that shows properties from the GeoJSON features and a column from the count_by_zona dataframe.
    folium.features.GeoJson(geojson_data,name=name,popup=folium.features.GeoJsonPopup(fields=['nomdis', 'nombre','count'])).add_to(choropleth_fig5) 

    '''
    Create a HeatMap layer using the latitud and longitud columns of the previously created 
    grouped dataframe and add it to a folium map object named 'madrid_map', along with a couple of TileLayer objects and a LayerControl object
    '''
    #
    ## Create the HeatMap layer and add it to the map object
    #HeatMap(data=df_grouped[['latitud', 'longitud']], radius=15).add_to(madrid_map)
    
    # Add a TileLayer with a dark theme to the map object
    folium.TileLayer('cartodbdark_matter',overlay=True,name="View in Dark Mode").add_to(madrid_map)
    
    # Add a TileLayer with a light theme to the map object
    folium.TileLayer('cartodbpositron',overlay=True,name="View in Light Mode").add_to(madrid_map)
    
    # Add a LayerControl object to the map object, which allows the user to toggle the display of different map layers 
    folium.LayerControl().add_to(madrid_map) 
    
    
    
    
    # Display the map
    folium_static(madrid_map,width=1400, height=1000)
