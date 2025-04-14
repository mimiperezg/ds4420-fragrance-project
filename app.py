import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import streamlit.components.v1 as components
from collab_filter_model import get_recommendations, generate_heatmap, generate_wordcloud



def home():
    st.title("Luxury Perfume Classification and Recommendation System")


    st.header("Introduction")
    st.write(
        """
        This project explores the intersection of **identity**, **aesthetics**, and **consumer behavior** in the world of perfume. 
        Using **machine learning**, our system classifies perfumes as **luxury** or **non-luxury** based on scent profiles and customer ratings, 
        while also offering personalized fragrance recommendations.
        """
)
     
    st.header("Methods")
    st.write(
        """
        We used two key machine learning models:
        1. **Multilayer Perceptron (MLP)** for classifying perfumes based on features like scent composition and customer ratings.
        2. **Collaborative Filtering** for recommending similar perfumes based on scent profiles.
        """
    )


    st.subheader("MLP Model for Luxury Classification")
    st.write(
        """
        The **MLP model** classifies perfumes as **luxury** or **non-luxury** using features like the perfume's scent composition, customer ratings, and price.
        We used a dataset with fragrance notes and accords, optimizing hyperparameters like learning rate and the number of neurons in the hidden layer.
        """
    )

 
    st.subheader("Collaborative Filtering for Fragrance Recommendations")
    st.write(
        """
        The **Collaborative Filtering** model recommends perfumes based on **similarity** in scent profiles. It uses the **cosine similarity** between **TF-IDF vectors** derived from fragrance notes and accords to find the most similar perfumes.
        """
    )


def collaborative_filtering():
    st.title("Collaborative Filtering Model")

   
    st.write(
        """
        ### Instructions:
        - Use **dashes (-)** instead of spaces in perfume names (e.g., `poppy-barley`).
        - Disregard the word "and" in perfume names if they exist (e.g., `poppy-and-barley` becomes `poppy-barley`
        - Brand names are **case-insensitive** (e.g., `Le-Labo` or `le-labo`).
        - For some names like `Chanel No 5`, write `chanel-n05`.
        """
    )

    # input
    perfume_name = st.text_input("Enter Perfume Name:")
    brand_name = st.text_input("Enter Brand Name:")
    top_n = st.slider("Number of Recommendations:", min_value=1, max_value=10, value=5)

    if st.button("Get Recommendations"):
        query_perfume, query_row, recommendations_df = get_recommendations(perfume_name, brand_name, top_n)

        if query_perfume is None:
            st.error(query_perfume)  # display error
        else:
            st.subheader("Recommendations")
            st.dataframe(recommendations_df)

            # heatmap
            generate_heatmap(query_row, recommendations_df)
            st.subheader("Cosine Similarity Heatmap")
            components.html(open("heatmap.html", "r").read(), height=600)

            # display word cloud
            generate_wordcloud(recommendations_df)
            st.image("wordcloud.png", caption="Common Notes in Recommendations")

def set_background_color(color):
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-color: {color};
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    

def main():

    set_background_color("#E6E6FA")  

    st.sidebar.title("Navigation")
    tab = st.sidebar.radio("Go to", ["Home", "Collaborative Filtering"])

    if tab == "Home":
        home()
    elif tab == "Collaborative Filtering":
        collaborative_filtering()

if __name__ == "__main__":
    main()
