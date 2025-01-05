import streamlit as st
import os

# Set up the Streamlit page
st.title("Navigation Interface")

st.write("Choose a level to proceed:")

# Buttons for navigation
if st.button("Level 0"):
    os.system("streamlit run level0.py")

if st.button("Level 1"):
    os.system("streamlit run level1.py")

if st.button("Level 2"):
    os.system("streamlit run level2.py")
