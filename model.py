# model.py

from tensorflow.keras.layers import (
    Input,
    Embedding,
    LSTM,
    Bidirectional,
    Concatenate,
    Dense,
    AdditiveAttention
)
from tensorflow.keras.models import Model


def build_model(vocab_size, embedding_dim=256, units=256):

    # -----------------------
    # Encoder
    # -----------------------
    encoder_inputs = Input(shape=(None,), name="encoder_inputs")
    enc_emb = Embedding(
        vocab_size,
        embedding_dim,
        name="encoder_embedding"
    )(encoder_inputs)

    encoder_lstm = Bidirectional(
        LSTM(
            units,
            return_sequences=True,
            return_state=True
        ),
        name="bidirectional_encoder"
    )

    enc_output, f_h, f_c, b_h, b_c = encoder_lstm(enc_emb)

    state_h = Concatenate(name="encoder_state_h")([f_h, b_h])
    state_c = Concatenate(name="encoder_state_c")([f_c, b_c])

    # -----------------------
    # Decoder
    # -----------------------
    decoder_inputs = Input(shape=(None,), name="decoder_inputs")

    dec_emb = Embedding(
        vocab_size,
        embedding_dim,
        name="decoder_embedding"
    )(decoder_inputs)

    decoder_lstm = LSTM(
        units * 2,
        return_sequences=True,
        return_state=True,
        name="decoder_lstm"
    )

    dec_output, _, _ = decoder_lstm(
        dec_emb,
        initial_state=[state_h, state_c]
    )

    # -----------------------
    # Attention
    # -----------------------
    attention = AdditiveAttention(name="attention_layer")
    attn_output = attention([dec_output, enc_output])

    concat = Concatenate(axis=-1, name="concat_layer")(
        [dec_output, attn_output]
    )

    output = Dense(
        vocab_size,
        activation="softmax",
        name="output_dense"
    )(concat)

    model = Model(
        [encoder_inputs, decoder_inputs],
        output,
        name="Seq2Seq_BiLSTM_Attention"
    )

    return model