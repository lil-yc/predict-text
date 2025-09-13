import streamlit as st
import os
import pickle
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# --------------- main.ipynb funcs ---------------

def _apply_temperature(logits, temperature=1.0):
    logits = np.asarray(logits, dtype=np.float64)
    if temperature <= 0:
        one_hot = np.zeros_like(logits)
        one_hot[np.argmax(logits)] = 1.0
        return one_hot
    logits = np.log(np.maximum(logits, 1e-9)) / float(temperature)
    exp = np.exp(logits - np.max(logits))
    return exp / np.sum(exp)

def suggest_next_words(model, tokenizer, seed_text, seq_len_minus1, top_k=5, temperature=1.0):
    seq = tokenizer.texts_to_sequences([seed_text])[0]
    seq = pad_sequences([seq], maxlen=seq_len_minus1, padding="pre")
    preds = model.predict(seq, verbose=0)[0]
    probs = _apply_temperature(preds, temperature=temperature)

    top_idx = probs.argsort()[-top_k:][::-1]
    idx2word = {idx: w for w, idx in tokenizer.word_index.items() if idx < (tokenizer.num_words or 10**9)}
    return [(idx2word.get(i, "<UNK>"), float(probs[i])) for i in top_idx]

def generate_text(model, tokenizer, seed_text, seq_len_minus1, num_words=10, temperature=1.0, greedy=False):
    out = seed_text.strip()
    vocab_limit = tokenizer.num_words or (len(tokenizer.word_index) + 1)
    idx2word = {idx: w for w, idx in tokenizer.word_index.items() if idx < vocab_limit}

    for _ in range(num_words):
        seq = tokenizer.texts_to_sequences([out])[0]
        seq = pad_sequences([seq], maxlen=seq_len_minus1, padding="pre")
        preds = model.predict(seq, verbose=0)[0]
        if greedy or temperature <= 0:
            next_id = int(np.argmax(preds))
        else:
            probs = _apply_temperature(preds, temperature=temperature)
            next_id = int(np.random.choice(len(probs), p=probs))
        next_word = idx2word.get(next_id, None)
        if not next_word:
            break
        out += " " + next_word
    return out

def load_artifacts(save_dir="predictive_text_artifacts"):
    model = tf.keras.models.load_model(os.path.join(save_dir, "model.keras"))
    with open(os.path.join(save_dir, "tokenizer.pkl"), "rb") as f:
        payload = pickle.load(f)
    tok = tokenizer_from_json(payload["config"])
    tok.num_words = payload["num_words"]
    with open(os.path.join(save_dir, "meta.json"), "r") as f:
        meta = json.load(f)
    return model, tok, meta



# --------------- streamlit ui ---------------

st.title("Predictive Text")
st.write("Enter a seed phrase to get next word suggestions or generate text.")

model, tokenizer, meta = load_artifacts()

seq_len_minus1 = meta.get("max_len", None)


seed = st.text_input("Seed text", value="There is a")
top_k = st.slider("Top K suggestions", min_value=1, max_value=10, value=5)
temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.8, step=0.05)
num_words = st.slider("Words to generate", min_value=1, max_value=20, value=6)


if st.button("Suggest next words"):
	
	st.write("### Top Suggestions:")
	

if st.button("Generate text"):
	
	st.write("### Generated Text:")
	