import streamlit as st
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model

# Load word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# Load pre-trained model
model = load_model("simple_rnn.h5")

# Function to decode
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# Function to preprocess user text
def preprocess_text(text):
    words = text.lower().split()
    encoded = [word_index.get(word, 2) + 3 for word in words]
    padded = sequence.pad_sequences([encoded], maxlen=500)
    return padded

# Streamlit UI
st.title("🎬 IMDB Movie Review Sentiment Analysis")
st.write("Enter a movie review to classify it as positive or negative.")

user_input = st.text_area("Movie Review")

if st.button("Classify"):
    processed = preprocess_text(user_input)
    prediction = model.predict(processed)
    sentiment = "Positive 😊" if prediction[0][0] > 0.5 else "Negative 😞"
    st.write(f"**Sentiment:** {sentiment}")
    st.write(f"**Confidence:** {prediction[0][0]:.4f}")
