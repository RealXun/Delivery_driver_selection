import streamlit as st
from streamlit_option_menu import option_menu
from scripts import drivers
from scripts import drivers_pedido
from PIL import Image

session = st.session_state
def delete_login_info():
    del st.session_state['login']
    del st.session_state['username']
    st.success("Logout successful")
    st.experimental_set_query_params(logout=True)
    session.clear()
    st.experimental_rerun()



def driver_menu(name):

    with st.sidebar:
        st.sidebar.image('logo.png', use_column_width=True) 
        st.write("Welcome ",name)
        choose = option_menu("Menú repartidores", ["Estado repartidor","Estado Pedidos","Logout"],
                            icons=['house','easel',"clipboard-data", '123',"graph-up", 'tv','person'],
                            menu_icon="cast", default_index=0,
                            styles={"container": {"padding": "5!important", "background-color": "#fafafa"},
                                    "icon": {"color": "orange", "font-size": "25px"}, 
                                    "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                                    "nav-link-selected": {"background-color": "#02ab21"},
                                    },
                            )
                            
        image = Image.open('logo.png')
        st.image(image)
        
    if choose == "Estado repartidor":
        drivers.drivers_info(name)
    if choose == "Estado Pedidos":
        drivers_pedido.drivers_pedido(name)
    elif choose == 'Logout':
        delete_login_info()
        
