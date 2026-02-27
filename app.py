# app.py

import os
import pickle
import numpy as np
import tensorflow as tf

from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.sequence import pad_sequences

# If using fallback weights loading
from model import build_model


# ----------------------------
# CONFIG
# ----------------------------
MAX_LEN = 40
MODEL_PATH = "models/seq2seq_model.keras"
WEIGHTS_PATH = "models/seq2seq_weights.weights.h5"
TOKENIZER_PATH = "models/tokenizer.pkl"

app = Flask(__name__)


# ----------------------------
# LOAD TOKENIZER
# ----------------------------
with open(TOKENIZER_PATH, "rb") as f:
    tokenizer = pickle.load(f)

vocab_size = len(tokenizer.word_index) + 1


# ----------------------------
# LOAD MODEL (SAFE LOADING)
# ----------------------------
try:
    # Preferred method
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Loaded full .keras model successfully.")

except Exception as e:
    print("Failed to load full model. Falling back to weights.")
    print("Error:", e)

    model = build_model(vocab_size)
    model.load_weights(WEIGHTS_PATH)
    print("Loaded model weights successfully.")


# ----------------------------
# TRANSLATION FUNCTION
# ----------------------------
def translate_text(text, src_lang, tgt_lang):

    # Match training format
    conditioned_input = f"<{src_lang}> <{tgt_lang}> {text}"

    # Convert to sequence
    input_seq = tokenizer.texts_to_sequences([conditioned_input])
    input_seq = pad_sequences(input_seq, maxlen=MAX_LEN, padding="post")

    # Prepare decoder input
    decoder_input = np.zeros((1, MAX_LEN))

    start_token = tokenizer.word_index.get("<start>")
    end_token = tokenizer.word_index.get("<end>")

    decoder_input[0, 0] = start_token

    output_words = []

    for i in range(1, MAX_LEN):

        predictions = model.predict(
            [input_seq, decoder_input],
            verbose=0
        )

        # Output shape:
        # (batch_size, sequence_length, vocab_size)

        predicted_id = np.argmax(predictions[0, i - 1])

        # Stop if <end>
        if predicted_id == end_token:
            break

        word = tokenizer.index_word.get(predicted_id, "")

        if word not in ["<start>", "<end>"]:
            output_words.append(word)

        # Feed predicted word back into decoder
        decoder_input[0, i] = predicted_id

    return " ".join(output_words)


# ----------------------------
# ROUTES
# ----------------------------
@app.route("/", methods=["GET", "POST"])
def index():

    result = ""

    if request.method == "POST":
        text = request.form["text"]
        src_lang = request.form["source"]
        tgt_lang = request.form["target"]

        result = translate_text(text, src_lang, tgt_lang)

    return render_template("index.html", result=result)


# ----------------------------
# RUN APP
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)