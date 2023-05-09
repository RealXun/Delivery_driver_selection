import streamlit as st
import os
from pathlib import Path

from streamlit_option_menu import option_menu



PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
scripts_folder = (PROJECT_ROOT + "/" + "scripts")
files_folder = (scripts_folder + "/" + "files")




def location_info():
    #Add the cover image for the cover page. Used a little trick to center the image
             # To display the header text using css style

    st.title('**ON Stream Location**')
    st.markdown(""" <style> .font {
        font-size:40px ; font-family: 'Cooper Black'; color: #FF9633;} 
        </style> """, unsafe_allow_html=True)
    st.write("We can see in real time the location of the drivers")
    st.markdown('<h1>Map of Available Drivers</h1>', unsafe_allow_html=True)


    

