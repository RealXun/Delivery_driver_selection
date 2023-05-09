import streamlit as st
from streamlit_option_menu import option_menu
from scripts import drivers
from scripts import pedidos_tienda
from scripts import formulas
from PIL import Image

session = st.session_state
def delete_login_info():
    del st.session_state['login']
    del st.session_state['username']
    st.success("Logout successful")
    st.experimental_set_query_params(logout=True)
    session.clear()
    st.experimental_rerun()


def tienda_menu(name,id):
    st.sidebar.image('logo.png', use_column_width=True)
    with st.sidebar:
        st.write("Bienvenido ",name)
        st.write("ID de comercio ",id)
        choose = option_menu("Tienda Menu", ["Pedidos","Logout"],
                            icons=['house','easel',"clipboard-data", '123',"graph-up", 'tv','person'],
                            menu_icon="cast", default_index=0,
                            styles={"container": {"padding": "5!important", "background-color": "#fafafa"},
                                    "icon": {"color": "orange", "font-size": "25px"}, 
                                    "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
                                    "nav-link-selected": {"background-color": "#02ab21"},
                                    },
                            )
    if choose == "Pedidos":
        pedidos_tienda.pedidos_tienda(name,id)
    elif choose == 'Logout':
        delete_login_info()
        
