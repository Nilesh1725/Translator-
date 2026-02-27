# train.py

import os
import pickle
import tensorflow as tf
import pandas as pd

from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from model import build_model


# -----------------------
# 1️⃣ GPU CHECK
# -----------------------
print("TensorFlow Version:", tf.__version__)
print("Available GPUs:", tf.config.list_physical_devices("GPU"))


# -----------------------
# 2️⃣ LOAD DATA
# -----------------------
df = pd.read_csv("translator_dataset.csv")

# Add language conditioning
df["input_text"] = (
    "<" + df["source_lang"] + "> "
    + "<" + df["target_lang"] + "> "
    + df["source_text"]
)

df["target_text"] = "<start> " + df["target_text"] + " <end>"


# -----------------------
# 3️⃣ TOKENIZER
# -----------------------
tokenizer = Tokenizer(filters='', oov_token="<unk>")
tokenizer.fit_on_texts(
    df["input_text"].tolist() +
    df["target_text"].tolist()
)

vocab_size = len(tokenizer.word_index) + 1
print("Vocabulary Size:", vocab_size)


# -----------------------
# 4️⃣ SEQUENCES
# -----------------------
input_seq = tokenizer.texts_to_sequences(df["input_text"])
target_seq = tokenizer.texts_to_sequences(df["target_text"])

MAX_LEN = 40

input_seq = pad_sequences(input_seq, maxlen=MAX_LEN, padding="post")
target_seq = pad_sequences(target_seq, maxlen=MAX_LEN, padding="post")


# -----------------------
# 5️⃣ TRAIN SPLIT
# -----------------------
X_train, X_val, y_train, y_val = train_test_split(
    input_seq,
    target_seq,
    test_size=0.1,
    random_state=42
)


# -----------------------
# 6️⃣ BUILD MODEL
# -----------------------
model = build_model(vocab_size)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# -----------------------
# 7️⃣ TRAIN (Teacher Forcing)
# -----------------------
history = model.fit(
    [X_train, y_train[:, :-1]],
    y_train[:, 1:],
    batch_size=64,
    epochs=20,
    validation_data=(
        [X_val, y_val[:, :-1]],
        y_val[:, 1:]
    )
)


# -----------------------
# 8️⃣ SAVE MODEL SAFELY
# -----------------------
os.makedirs("models", exist_ok=True)

# Save weights (most stable)
model.save_weights("models/seq2seq_weights.h5")

# Save full model (modern format)
model.save("models/seq2seq_model.keras")

# Save tokenizer
with open("models/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

print("Training complete. Model saved successfully.")