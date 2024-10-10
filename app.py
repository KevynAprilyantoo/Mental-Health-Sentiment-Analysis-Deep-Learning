import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import load_model
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer

import nltk

st.image("dataset-cover.jpg", use_column_width=True)

page_bg_img = '''
<style>
body {
    background-color: #87CEEB; 
    color: #000000; 
}

</style>
'''

# Menyisipkan CSS ke dalam Streamlit
st.markdown(page_bg_img, unsafe_allow_html=True)

# Ensure you have the required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Memuat model Keras yang sudah disimpan
model = load_model('sentiment_analysis_model.h5')

import pickle

# Memuat tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

classes = np.array(['Anxiety', 'Normal', 'Depression', 'Suicidal', 'Stress', 'Bipolar', 'Personality disorder'])  # Change according to your model

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()
    # Tokenize
    word_list = word_tokenize(text)
    # Remove stopwords
    eng_stopwords = stopwords.words('english')
    word_list = [word for word in word_list if word not in eng_stopwords]
    # Stemming
    stemmer = SnowballStemmer('english')
    word_list = [stemmer.stem(word) for word in word_list]
    # Join words back to a single string
    return ' '.join(word_list)

# Function to predict sentiment
def predict_sentiment(text):
    processed_text = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([processed_text])
    padded = pad_sequences(seq, maxlen=200)  # Adjust maxlen based on your model
    pred = model.predict(padded)
    return classes[np.argmax(pred)]

# def predict_sentiment(text):
#     processed_text = preprocess_text(text)
    
#     # Debug: print tokenized sequence
#     st.write("Processed Text: ", processed_text)
    
#     seq = tokenizer.texts_to_sequences([processed_text])
    
#     # Debug: print tokenized sequence
#     st.write("Tokenized Sequence: ", seq)
    
#     padded = pad_sequences(seq, maxlen=400)  # Adjust maxlen based on your model
    
#     # Debug: print padded sequence
#     st.write("Padded Sequence: ", padded)
    
#     pred = model.predict(padded)
    
#     # Debug: print prediction output
#     st.write("Prediction Raw Output: ", pred)
    
#     return classes[np.argmax(pred)]

st.title("Mental Health Prediction - Sentiment Analysis WebApp.") 

st.write("Masukkan teks di bawah ini untuk menganalisis sentimen.")  

user_input = st.text_area("Please Enter your text")

if st.button("Prediksi Sentimen"):
    if user_input:
        sentiment = predict_sentiment(user_input)
        st.write(f"Sentimen: {sentiment}")
    else:
        st.write("Masukkan teks terlebih dahulu.")

