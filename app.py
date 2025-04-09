import streamlit as st
from app_pages import home, visual_study, predictor, performance, hypothesis
from src.data_management import download_sample_dataset

# Ensure sample dataset is downloaded (for Heroku)
download_sample_dataset()

# Set up main navigation
st.set_page_config(page_title="Cherry Leaf Mildew Detection", layout="wide")

# Sidebar menu
pages = {
    "Home": home,
    "Visual Differentiation Study": visual_study,
    "Mildew Detection Tool": predictor,
    "Model Performance": performance,
    "Hypothesis & Validation": hypothesis
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(pages.keys()))

# Render selected page
pages[selection].app()  # Make sure all modules have an `app()` function