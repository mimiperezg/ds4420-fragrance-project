# The Fragrance Aesthetic: Fragrance Classification and Recommendation

### Live App: https://fragrance-ml.streamlit.app/


| File/Folder              | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| `app.py`                 | Main Streamlit app (deployed online)                                        |
| `collab_filter.ipynb`    | Python code for content-based collaborative filtering                       |
| `mlp_model.Rmd`          | R Markdown Multilayer Perceptron (MLP) classifier to predict luxury vs. budget perfumes |
| `preprocess.ipynb`       | Data preprocessing, cleaning, merging Fragrantica and eBay datasets         |
| `web_scrape.ipynb`       | Selenium manual web scraper for extracting Fragrantica data                 


## Project Objectives
- **Supervised Classification**: Develop MLP model to classify luxury vs. non-luxury perfumes based on fragrance composition and customer ratings  
- **Content-Based Recommendation**: Implement a recommendation system using TF-IDF and cosine similarity to identify perfumes with semantically similar note profiles

## Data Collection & Preprocessing
- Fragrantica Dataset: Through Kaggle, and some web-scraped using Selenium, containing scent notes, main accords, ratings, and brand names
- eBay Dataset: Kaggle dataset with resale pricing data used to infer luxury vs. budget classification


## Contributors
Mimi Perez (@mimiperezg)
Angela Wu (@angw100)

## Acknowledgements
This project was developed as part of DS 4420: Machine Learning at Northeastern University. Data was obtained from publicly available sources (Fragrantica, eBay) for academic purposes.
