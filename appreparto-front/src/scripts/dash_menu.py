import streamlit as st
from streamlit_option_menu import option_menu
from scripts import dashboard
from scripts import shops
from scripts import heatmaps
from scripts import drivers_status
from scripts import pedidos_menu
from PIL import Image

session = st.session_state
def delete_login_info():
    del st.session_state['login']
    del st.session_state['username']
    st.success("Logout successful")
    st.experimental_set_query_params(logout=True)
    session.clear()
    st.experimental_rerun()



def dash_menu(name):

    with st.sidebar:


        st.sidebar.image('logo.png', use_column_width=True)
        st.write("Bienvenido ",name)
        # If you do not want to display an opcion from the menu, just erase it and change the below options
        choose = option_menu("Menú Admin", ["Comercios","Repartidores","Mapas de calor","Pedidos","Logout"],
                            icons=['house','easel',"clipboard-data", '123',"graph-up", 'tv','person'],
                            menu_icon="cast", default_index=0,
                            styles={"container": {"padding": "5!important", "background-color": "#fafafa"},
                                    "icon": {"color": "orange", "font-size": "25px"}, 
                                    "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                                    "nav-link-selected": {"background-color": "#02ab21"},
                                    },
                            )
    if choose == "Comercios":
        shops.shops_info(name)# Possible option to display a map of the shops
    elif choose == "Repartidores":
        drivers_status.drivers_status(name)# Possible option to display a map of Available Drives
    elif choose == "Mapas de calor":
        heatmaps.heatmaps(name)# Possible option to display a heatmap  with real time hot areas
    elif choose == "Pedidos":
        pedidos_menu.pedidos(name)# Possible option to display a heatmap  with real time hot areas
    elif choose == 'Logout':
        delete_login_info()