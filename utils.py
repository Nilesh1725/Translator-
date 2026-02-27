import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

MAX_LEN = 40

def preprocess_input(text, source_lang):
    return f"{source_lang} {text}"

def encode_text(tokenizer, text):
    seq = tokenizer.texts_to_sequences([text])
    return pad_sequences(seq, maxlen=MAX_LEN, padding='post')

def greedy_decode(model, tokenizer, input_seq):
    output_seq = np.zeros((1, MAX_LEN))
    output_seq[0, 0] = tokenizer.word_index["<start>"]

    for i in range(1, MAX_LEN):
        preds = model.predict([input_seq, output_seq], verbose=0)
        token_id = np.argmax(preds[0, i-1])
        output_seq[0, i] = token_id

        if token_id == tokenizer.word_index.get("<end>"):
            break

    return decode_sequence(tokenizer, output_seq[0])

def decode_sequence(tokenizer, sequence):
    words = []
    for idx in sequence:
        word = tokenizer.index_word.get(int(idx))
        if word not in ["<start>", "<end>", None]:
            words.append(word)
    return " ".join(words)