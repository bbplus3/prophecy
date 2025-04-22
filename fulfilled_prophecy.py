import numpy as np
import pandas as pd
import warnings
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import streamlit as st
import pickle
from sklearn.decomposition import TruncatedSVD

# streamlit run fulfilled_prophecy.py

# Suppress warnings
warnings.filterwarnings('ignore')

# Download necessary NLTK resources
#nltk.download('stopwords')

stop_words = ['all', 'to', 'any', "he'd", "we've", 'this', 'have', 'whom', "isn't", "wasn't", 'own', 'now', 'do', "mightn't", 'but', 'yourselves', 'under', "i've", 'his', 'is', "haven't", 'over', 'doesn', "he's", 'her', 'your', "you've", 'each', 'the', "she'll", 'did', "you'll", 'until', "wouldn't", 'during', 'some', 'he', 'than', "didn't", 'then', 'with', 'had', "it's", 'and', "should've", 'few', "it'll", 'there', 'which', 'why', "we're", 'should', 'other', "i'll", 'an', 'been', 'herself', "needn't", 'above', "hasn't", 'both', 'will', 'only', "we'll", 'before', 'here', "we'd", 'again', 'what', "you'd", "shouldn't", 'has', 'me', "i'd", 'were', "aren't", 'so', "she's", "hadn't", 'she', 'o', 'from', 'on', 'ours', "they've", 'very', "don't", 'down', 'further', 'it', 'by', 'once', 'if', 'doing', 'are', 'no', 'i', 'through', 'yours', 'about', "she'd", 'most', 'how', "mustn't", 'as', 'myself', 'being', 'their', 'was', 'between', 'or', 'into', 'when', 'them', "they're", 'him', "couldn't", 'shouldn', 'who', 'my', "doesn't", 'where', 'at', 'off', 'yourself', 'for', 'its', "won't", 'such', "he'll", 'hers', 'be', 'after', 'not', 'same', 'these', 'that', 'below', "shan't", "they'll", 'nor', 'they', 'having', 'too', 'himself', 'those', 'out', "i'm", 'itself', 'just', 'while', 'does', "that'll", 'theirs', "they'd", 'in', 'can', 'of', 'am', 'because', "it'd", 'more', 'you', "weren't", 'we', 'themselves', 'ourselves', 'a', "you're", 'up', 'our', 'against']

# Streamlit Page Configuration
st.set_page_config(page_title="Prophecy Fulfillment", layout="wide")
##################################################################################################################
# Cache Data Loading
@st.cache_data
def load_prophets():
    prophets = pd.read_csv("t_bbe.csv").dropna()
    
    # Map book numbers to book names
    prophets_names = {
        1: 'Genesis', 2: 'Exodus', 3: 'Leviticus', 4: 'Numbers', 5: 'Deuteronomy',
        6: 'Joshua', 7: 'Judges', 8: 'Ruth', 9: '1 Samuel', 10: '2 Samuel',
        11: '1 Kings', 12: '2 Kings', 13: '1 Chronicles', 14: '2 Chronicles',
        15: 'Ezra', 16: 'Nehemiah', 17: 'Esther', 18: 'Job', 19: 'Psalms',
        20: 'Proverbs', 21: 'Ecclesiastes', 22: 'Song of Solomon', 23: 'Isaiah',
        24: 'Jeremiah', 25: 'Lamentations', 26: 'Ezekiel', 27: 'Daniel',
        28: 'Hosea', 29: 'Joel', 30: 'Amos', 31: 'Obadiah', 32: 'Jonah',
        33: 'Micah', 34: 'Nahum', 35: 'Habakkuk', 36: 'Zephaniah', 37: 'Haggai',
        38: 'Zechariah', 39: 'Malachi'
    }
    # Map book numbers to names
    prophets['Book Name'] = prophets['b'].map(prophets_names)

    # Process text: Remove stopwords
    #stop_words = set(stopwords.words('english'))
    prophets['corpus'] = prophets['t'].astype(str).str.lower().apply(
        lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    return prophets, prophets_names

# Load data
prophets, prophets_names = load_prophets()
##################################################################################################################
# Cache Data Loading
@st.cache_data
def load_fulfilled():
    fulfilled = pd.read_csv("t_bbe.csv").dropna()
    
    # Map book numbers to book names
    fulfilled_names = {
        40: 'Matthew', 41: 'Mark', 42: 'Luke',
        43: 'John', 44: 'Acts', 45: 'Romans', 46: '1 Corinthians',
        47: '2 Corinthians', 48: 'Galatians', 49: 'Ephesians', 50: 'Philippians',
        51: 'Colossians', 52: '1 Thessalonians', 53: '2 Thessalonians',
        54: '1 Timothy', 55: '2 Timothy', 56: 'Titus', 57: 'Philemon',
        58: 'Hebrews', 59: 'James', 60: '1 Peter', 61: '2 Peter',
        62: '1 John', 63: '2 John', 64: '3 John', 65: 'Jude', 66: 'Revelation'
    }
    # Map book numbers to names
    fulfilled['Book Name'] = fulfilled['b'].map(fulfilled_names)

    # Process text: Remove stopwords
    #stop_words = set(stopwords.words('english'))
    fulfilled['corpus'] = fulfilled['t'].astype(str).str.lower().apply(
        lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

    return fulfilled, fulfilled_names

# Load data
fulfilled, fulfilled_names = load_fulfilled()
##################################################################################################################
# Compute TF-IDF & Cosine Similarity
@st.cache_resource
def compute_similarity():
    vectorizer = TfidfVectorizer()
    tf_idf_matrix = vectorizer.fit_transform(fulfilled['corpus'])
    svd = TruncatedSVD(n_components=100)  # Reduce to 100 dimensions
    tfidf_reduced = svd.fit_transform(tf_idf_matrix)
    return cosine_similarity(tfidf_reduced)

similarity_matrix = compute_similarity()

# Reverse book name lookup
prophets_book_numbers = {v: k for k, v in prophets_names.items()}
fulfilled_book_numbers = {v: k for k, v in fulfilled_names.items()}

# **Find Similar Verses Function**
def top_verse(input_book, input_chapter, input_verse, top_n=10):
    try:
        book_num = str(prophets_book_numbers.get(input_book, ""))
        locator = prophets.loc[
            (prophets['b'].astype(str) == book_num) &
            (prophets['c'].astype(str) == str(input_chapter)) &
            (prophets['v'].astype(str) == str(input_verse))
        ]
        if locator.empty:
            return pd.DataFrame(columns=["Book", "Chapter", "Verse", "Text", "Similarity Score"])
        idx = locator.index[0]
        similarity_scores = list(enumerate(similarity_matrix[idx]))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        sim_indices = [i[0] for i in similarity_scores[1:top_n + 1]]
        sim_values = [i[1] for i in similarity_scores[1:top_n + 1]]
        recommended = fulfilled.iloc[sim_indices].copy()
        recommended['Similarity Score'] = sim_values
        recommended = recommended[['Book Name', 'c', 'v', 't', 'Similarity Score']]
        recommended.columns = ["Book", "Chapter", "Verse", "Text", "Similarity Score"]
        recommended = recommended[recommended['Book'].notna()]
        return recommended
    except Exception as e:
        st.error(f"Error in recommendation: {e}")
        return pd.DataFrame(columns=["Book", "Chapter", "Verse", "Text", "Similarity Score"])

##################################################################################################################

st.title("📖 Prophecy Verse Search")
st.info("Enter a Book, Chapter and Verse ➡️ click 'Find Prophecy Fulfillment Verses' to find New Testament Bible verses where Old Testament Prophecies were fulfilled.")
st.markdown("[Review Prophecies and Corresponding Fullfillment Verses](https://www.jesusfilm.org/blog/old-testament-prophecies/)")

with st.form("user_input"):
    col1, col2, col3 = st.columns(3)
    with col1:
        input_book = st.selectbox("Select Book", prophets_names.values())
    with col2:
        input_chapter = st.number_input("Chapter", min_value=1, max_value=150, value=1, step=1)
    with col3:
        input_verse = st.number_input("Verse", min_value=1, max_value=176, value=1, step=1)

    #top_n = st.slider("Number of Verses to Examine", min_value=1, max_value=50, value=10, step=5)

    submitted = st.form_submit_button("➡️Find Prophecy Fulfillment Verses")

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

    #st.dataframe(results.style.set_properties(subset=['Text'], **{'white-space': 'normal'}))
