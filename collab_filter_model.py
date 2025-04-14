import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import plotly.graph_objects as go
import re


df = pd.read_csv("data/full_perfumes.csv")
df = df.dropna(subset=['notes', 'mainaccord1'])
df['combined_features'] = df[['mainaccord1', 'mainaccord2', 'mainaccord3', 'mainaccord4', 'notes']].fillna('').agg(' '.join, axis=1)

# tokenizer and cleaning functions
def clean_token_string(text):
    text = re.sub(r'\b(and)\b', '', text.lower())
    text = re.sub(r'-+', '-', text)
    text = re.sub(r'-$', '', text)
    text = re.sub(r'^-', '', text)
    return text.strip()

def tokenize(row):
    accords = [clean_token_string(acc) for acc in [
        row['mainaccord1'], row['mainaccord2'], row['mainaccord3'], row['mainaccord4']
    ] if pd.notna(acc)]
    notes = [clean_token_string(note) for note in row['notes'].split(',') if note.strip()]
    return accords + notes

df['tokens'] = df.apply(tokenize, axis=1)

# build vocab
vocab = {}
for tokens in df['tokens']:
    for token in set(tokens):
        vocab[token] = vocab.get(token, 0) + 1

total_docs = len(df)
idf = {word: np.log(total_docs / df_count) for word, df_count in vocab.items()}

# compute TF-IDF vectors
def compute_tfidf(tokens, idf):
    tf = {}
    for token in tokens:
        tf[token] = tf.get(token, 0) + 1
    max_tf = max(tf.values())
    return {token: (count / max_tf) * idf[token] for token, count in tf.items() if token in idf}

df['tfidf'] = df['tokens'].apply(lambda tokens: compute_tfidf(tokens, idf))

# similarity
def cosine_similarity_sparse(vec1, vec2):
    common = set(vec1.keys()) & set(vec2.keys())
    dot_product = sum(vec1[t] * vec2[t] for t in common)
    norm1 = np.sqrt(sum(v**2 for v in vec1.values()))
    norm2 = np.sqrt(sum(v**2 for v in vec2.values()))
    return dot_product / (norm1 * norm2) if norm1 and norm2 else 0.0

# prepare for lookup
df['perfume_brand_key'] = (df['perfume'].str.lower() + '|' + df['brand'].str.lower()).str.strip()
indices = pd.Series(df.index, index=df['perfume_brand_key']).drop_duplicates()

# recommendation
def get_recommendations(perfume_name, brand_name, top_n=5):
    query_key = f"{perfume_name.lower()}|{brand_name.lower()}"
    idx = indices.get(query_key)
    if idx is None:
        return (
            f"⚠️ No perfume found for: '{perfume_name}' by '{brand_name}'\n"
            "Make sure to:\n"
            "- Use dashes instead of spaces (e.g., poppy-barley)\n"
            "- Check for brand spelling\n"
            "- For some names like 'Chanel No 5', use 'n05'\n"
        ), None, None

    query_vec = df.loc[idx, 'tfidf']
    query_perfume = df.loc[idx, 'perfume']

    similarities = []
    for i, row in df.iterrows():
        if i == idx:
            continue
        sim = cosine_similarity_sparse(query_vec, row['tfidf'])
        similarities.append({
            "Perfume": row['perfume'],
            "Brand": row['brand'],
            "Similarity": round(sim, 3),
            "Rating": row['rating_value'],
            "Gender": row['gender'],
            "Main Accords": ', '.join([str(row.get(f'mainaccord{j}', '')) for j in range(1, 5) if pd.notna(row.get(f'mainaccord{j}', None))]),
            "Notes": row['notes']
        })

    similarities.sort(key=lambda x: x["Similarity"], reverse=True)
    recommendations = pd.DataFrame(similarities[:top_n])
    return query_perfume, df.loc[idx], recommendations

# generating wordcloud
def generate_wordcloud(recommendations_df):
    all_notes = ', '.join(recommendations_df['Notes'].tolist())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_notes)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title("Common Notes in Top Recommendations")
    plt.tight_layout()
    plt.savefig("wordcloud.png") 
    plt.close()


def generate_heatmap(query_perfume, recommendations_df):

    perfumes = [query_perfume['perfume']] + recommendations_df['Perfume'].tolist()
    vectors = [df.loc[indices[query_perfume['perfume_brand_key']], 'tfidf']] + \
              [df.loc[indices[f"{row['Perfume'].lower()}|{row['Brand'].lower()}"], 'tfidf'] for _, row in recommendations_df.iterrows()]
    sim_matrix = np.array([[cosine_similarity_sparse(v1, v2) for v2 in vectors] for v1 in vectors])

    # hover text for stats
    hover_text = []
    for i, perfume1 in enumerate(perfumes):
        hover_row = []
        for j, perfume2 in enumerate(perfumes):
            if i == 0:  # query perfume
                stats = f"Perfume: {perfume2}<br>Brand: {query_perfume['brand']}<br>Similarity: {sim_matrix[i][j]:.3f}"
            else:  # recommended perfumes
                stats = f"Perfume: {perfume2}<br>Brand: {recommendations_df.iloc[j-1]['Brand']}<br>Similarity: {sim_matrix[i][j]:.3f}<br>Rating: {recommendations_df.iloc[j-1]['Rating']}<br>Notes: {recommendations_df.iloc[j-1]['Notes']}"
            hover_row.append(stats)
        hover_text.append(hover_row)


    fig = go.Figure(data=go.Heatmap(
        z=sim_matrix,
        x=perfumes,
        y=perfumes,
        text=hover_text,
        hoverinfo="text",
        colorscale="Viridis",
        showscale=True
    ))

    for i in range(len(perfumes)):
        for j in range(len(perfumes)):
            fig.add_annotation(
                x=perfumes[j],
                y=perfumes[i],
                text=f"{sim_matrix[i][j]:.2f}",
                showarrow=False,
                font=dict(color="white" if sim_matrix[i][j] > 0.5 else "black")  
            )


    fig.update_layout(
        title="Cosine Similarity Heatmap",
        xaxis=dict(title="Perfumes", tickangle=45, automargin=True),
        yaxis=dict(title="Perfumes", automargin=True),
        autosize=True,
        margin=dict(l=100, r=100, t=100, b=100)  # preventing overlap
    )

    fig.write_html("heatmap.html")