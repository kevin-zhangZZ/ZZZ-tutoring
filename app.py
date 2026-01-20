"""
Streamlit app for Monte Carlo simulations.

This is the home page for the multipage app. Use the sidebar page
selector to open each simulation.
"""

import streamlit as st

st.set_page_config(
    page_title="Monte Carlo Simulations",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Monte Carlo Simulations")

st.markdown(
    "Welcome! This app contains interactive Monte Carlo demos "
    "for teaching and exploring probability and estimation."
)

st.markdown("#### Available simulation pages")
st.markdown(
    "- **Area under curve (sin(πx) on [0,1])** – estimate the area under the curve using random points.\n"
    "- **Buffon's Needle (π estimation)** – estimate π from random needle drops on parallel lines."
)

st.info(
    "Use the **page selector in the sidebar** to switch between simulations. "
    "Each page has its own controls and visualizations."
)

