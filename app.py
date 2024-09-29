import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

# Load model and tokenizer
model = load_model("LSTMmodel.h5")
with open("tokenizer.pkl", 'rb') as file:
    tokenizer = pickle.load(file)

# Page title
st.set_page_config(page_title="Hamlet Next Word Predictor", layout="centered")
st.title("🔮 Hamlet Next Word Predictor")
st.write("Welcome! This tool predicts the next word in a given line from Shakespeare's Hamlet.")

# Input section with better labeling
st.subheader("Enter a line of text:")
input_text = st.text_input("Write your line here", placeholder="E.g., To be or not to...")

# Prediction section
if input_text:
    token_list = tokenizer.texts_to_sequences([input_text])[0]
    max_seq_length = 14

    if len(token_list) > max_seq_length:
        token_list = token_list[-(max_seq_length):]

    padded_list = pad_sequences([token_list], maxlen=max_seq_length)

    # Make prediction
    predicted = model.predict(padded_list)
    predicted_word_index = np.argmax(predicted, axis=1)

    # Find the predicted word
    predicted_word = None
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            predicted_word = word
            break

    # Display the predicted word
    if predicted_word:
        st.success(f"✨ The next word might be: **{predicted_word}**")
    else:
        st.error("Could not predict the next word. Please try a different input.")
else:
    st.info("Please enter a line from Hamlet to predict the next word.")

# Footer
st.write("Made with ❤️ by Prajwal Dumbre")
