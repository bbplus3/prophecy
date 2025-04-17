import os
import joblib
import numpy as np
import pandas as pd
import nltk
import warnings
import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords

# Suppress warnings
warnings.filterwarnings("ignore")

# Download NLTK stopwords
#nltk.download("stopwords")
stop_words = ['all', 'to', 'any', "he'd", "we've", 'this', 'have', 'whom', "isn't", "wasn't", 'own', 'now', 'do', "mightn't", 'but', 'yourselves', 'under', "i've", 'his', 'is', "haven't", 'over', 'doesn', "he's", 'her', 'your', "you've", 'each', 'the', "she'll", 'did', "you'll", 'until', "wouldn't", 'during', 'some', 'he', 'than', "didn't", 'then', 'with', 'had', "it's", 'and', "should've", 'few', "it'll", 'there', 'which', 'why', "we're", 'should', 'other', "i'll", 'an', 'been', 'herself', "needn't", 'above', "hasn't", 'both', 'will', 'only', "we'll", 'before', 'here', "we'd", 'again', 'what', "you'd", "shouldn't", 'has', 'me', "i'd", 'were', "aren't", 'so', "she's", "hadn't", 'she', 'o', 'from', 'on', 'ours', "they've", 'very', "don't", 'down', 'further', 'it', 'by', 'once', 'if', 'doing', 'are', 'no', 'i', 'through', 'yours', 'about', "she'd", 'most', 'how', "mustn't", 'as', 'myself', 'being', 'their', 'was', 'between', 'or', 'into', 'when', 'them', "they're", 'him', "couldn't", 'shouldn', 'who', 'my', "doesn't", 'where', 'at', 'off', 'yourself', 'for', 'its', "won't", 'such', "he'll", 'hers', 'be', 'after', 'not', 'same', 'these', 'that', 'below', "shan't", "they'll", 'nor', 'they', 'having', 'too', 'himself', 'those', 'out', "i'm", 'itself', 'just', 'while', 'does', "that'll", 'theirs', "they'd", 'in', 'can', 'of', 'am', 'because', "it'd", 'more', 'you', "weren't", 'we', 'themselves', 'ourselves', 'a', "you're", 'up', 'our', 'against']

# Streamlit page setup
st.set_page_config(page_title="Prophecy Fulfillment", layout="wide")

# Load and clean data
@st.cache_data

def load_data(file_path, book_map):
    df = pd.read_csv(file_path).dropna()
    df['Book Name'] = df['b'].map(book_map)
    #stop_words = set(stopwords.words('english'))
    df['corpus'] = df['t'].astype(str).str.lower().apply(
        lambda x: ' '.join([word for word in x.split() if word not in stop_words])
    )
    return df

# Old Testament
prophets_map = {i: name for i, name in enumerate([
    'Genesis', 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy',
    'Joshua', 'Judges', 'Ruth', '1 Samuel', '2 Samuel',
    '1 Kings', '2 Kings', '1 Chronicles', '2 Chronicles', 'Ezra',
    'Nehemiah', 'Esther', 'Job', 'Psalms', 'Proverbs',
    'Ecclesiastes', 'Song of Solomon', 'Isaiah', 'Jeremiah',
    'Lamentations', 'Ezekiel', 'Daniel', 'Hosea', 'Joel',
    'Amos', 'Obadiah', 'Jonah', 'Micah', 'Nahum',
    'Habakkuk', 'Zephaniah', 'Haggai', 'Zechariah', 'Malachi'
], start=1)}

# New Testament
fulfilled_map = {i: name for i, name in enumerate([
    'Matthew', 'Mark', 'Luke', 'John', 'Acts', 'Romans',
    '1 Corinthians', '2 Corinthians', 'Galatians', 'Ephesians', 'Philippians',
    'Colossians', '1 Thessalonians', '2 Thessalonians', '1 Timothy', '2 Timothy',
    'Titus', 'Philemon', 'Hebrews', 'James', '1 Peter', '2 Peter',
    '1 John', '2 John', '3 John', 'Jude', 'Revelation'
], start=40)}

prophets = load_data("t_bbe.csv", prophets_map)
fulfilled = load_data("t_bbe.csv", fulfilled_map)

# TF-IDF Similarity with joblib caching
@st.cache_resource
def compute_similarity():
    vec_path = "tfidf_vectorizer.joblib"
    mat_path = "tfidf_matrix.joblib"

    if os.path.exists(vec_path) and os.path.exists(mat_path):
        vectorizer = joblib.load(vec_path)
        tf_idf_matrix = joblib.load(mat_path)
    else:
        vectorizer = TfidfVectorizer()
        tf_idf_matrix = vectorizer.fit_transform(fulfilled['corpus'])
        joblib.dump(vectorizer, vec_path)
        joblib.dump(tf_idf_matrix, mat_path)

    return cosine_similarity(tf_idf_matrix)

similarity_matrix = compute_similarity()

# Reverse book maps
prophets_num = {v: k for k, v in prophets_map.items()}
fulfilled_num = {v: k for k, v in fulfilled_map.items()}

# Find top similar verses

def top_verse(book, chapter, verse, top_n=10):
    try:
        book_num = prophets_num.get(book)
        locator = prophets[(prophets['b'] == book_num) &
                           (prophets['c'] == chapter) &
                           (prophets['v'] == verse)]

        if locator.empty:
            return pd.DataFrame(columns=["Book", "Chapter", "Verse", "Text", "Similarity Score"])

        idx = locator.index[0]
        scores = list(enumerate(similarity_matrix[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        sim_indices = [i[0] for i in scores[1:top_n+1]]
        sim_values = [i[1] for i in scores[1:top_n+1]]

        recommended = fulfilled.iloc[sim_indices].copy()
        recommended['Similarity Score'] = sim_values
        recommended = recommended[['Book Name', 'c', 'v', 't', 'Similarity Score']]
        recommended.columns = ["Book", "Chapter", "Verse", "Text", "Similarity Score"]
        recommended = recommended[recommended['Book'].notna()]

        return recommended
    except Exception as e:
        st.error(f"Error: {e}")
        return pd.DataFrame(columns=["Book", "Chapter", "Verse", "Text", "Similarity Score"])

# Streamlit UI
st.title("📖 Prophecy Verse Search")
st.info("Enter a Book, Chapter and Verse ➡️ click 'Find Prophecy Fulfillment Verses' to find New Testament verses fulfilling Old Testament prophecies.")
st.markdown("[Reference: 351 OT Prophecies Fulfilled in NT](https://www.jesusfilm.org/blog/old-testament-prophecies/)")

with st.form("verse_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        input_book = st.selectbox("Book", prophets_map.values())
    with col2:
        input_chapter = st.number_input("Chapter", min_value=1, max_value=150, value=1)
    with col3:
        input_verse = st.number_input("Verse", min_value=1, max_value=176, value=1)

    #top_n = st.slider("Number of Results", 1, 50, 10)
    submitted = st.form_submit_button("➡️ Find Prophecy Fulfillment Verses")

if submitted:
    results = top_verse(input_book, input_chapter, input_verse, top_n=10)
    searched_verse = prophets.loc[
        (prophets['Book Name'] == input_book) &
        (prophets['c'].astype(str) == str(input_chapter)) &
        (prophets['v'].astype(str) == str(input_verse))
    ]

    if not searched_verse.empty:
        st.write(f"**Input Verse:** {searched_verse.iloc[0]['t']}")
        st.write("### 🔍 Corresponding Verses:")
        for i, row in results.iterrows():
            st.write(f"**Book:** {row['Book']} | **Chapter:** {row['Chapter']} | **Verse:** {row['Verse']}")
            st.write(f"**Text:** {row['Text']} (Similarity: {row['Similarity Score']:.2f})")
    else:
        st.write("Verse not found.")
