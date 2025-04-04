# chatbot_web.py

import pandas as pd
import streamlit as st
import os
from PIL import Image
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from datetime import datetime
import csv
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel


# ========= CONFIG =========
LOG_FILE = "chat_logs.csv"
MAX_LEN = 100

# ========= STREAMLIT CONFIG =========
st.set_page_config(page_title="GoStop BBQ Recommender", layout="centered")

with st.sidebar:
    if os.path.exists("gostop.jpeg"):
        logo = Image.open("gostop.jpeg")
        st.image(logo, use_container_width=True)
    else:
        st.warning("⚠️ Logo 'gostop.jpeg' not found.")

    if st.button("🗑️ Clear Chat", key="clear"):
        st.session_state.chat_history = []
        st.rerun()

# ========= MENU DATA =========
menu_actual = [
    "soondubu jjigae", "prawn soondubu jjigae", "kimchi jjigae", "tofu jjigae",
    "samgyeopsal", "spicy samgyeopsal", "woo samgyup", "spicy woo samgyup",
    "bulgogi", "dak bulgogi", "spicy dak bulgogi", "meltique tenderloin", "odeng",
    "beef soondubu jjigae", "pork soondubu jjigae"
]

menu_categories = {
    "spicy": ["spicy samgyeopsal", "spicy woo samgyup", "spicy dak bulgogi", "kimchi jjigae", "budae jjigae"],
    "meat": ["samgyeopsal", "woo samgyup", "bulgogi", "dak bulgogi", "saeng galbi", "meltique tenderloin"],
    "soup": ["kimchi jjigae", "tofu jjigae", "budae jjigae", "soondubu jjigae", "beef soondubu jjigae", "pork soondubu jjigae", "prawn soondubu jjigae"],
    "seafood": ["prawn soondubu jjigae", "odeng"],
    "beef": ["bulgogi", "beef soondubu jjigae", "meltique tenderloin"],
    "pork": ["samgyeopsal", "spicy samgyeopsal", "pork soondubu jjigae"],
    "bbq": ["samgyeopsal", "woo samgyup", "bulgogi"],
    "non_spicy": ["tofu jjigae", "soondubu jjigae", "beef soondubu jjigae", "meltique tenderloin", "odeng"],
    "tofu_based": ["tofu jjigae", "soondubu jjigae", "beef soondubu jjigae", "pork soondubu jjigae"]
}

keyword_aliases = {
    "non spicy": "non_spicy", "non-spicy": "non_spicy", "not spicy": "non_spicy", "mild": "non_spicy",
    "grill": "bbq", "barbecue": "bbq", "bbq": "bbq",
    "hot soup": "soup", "warm soup": "soup",
    "hot": "spicy", "spicy": "spicy",
    "soup": "soup", "broth": "soup", "jjigae": "soup",
    "fish": "seafood", "prawn": "seafood", "seafood": "seafood",
    "beef": "beef", "pork": "pork", "meat": "meat",
    "tofu": "tofu_based"
}

menu_aliases = {
    "soondubu": "soondubu jjigae",
    "suundobu": "soondubu jjigae",
    "beef soondubu": "beef soondubu jjigae",
    "pork soondubu": "pork soondubu jjigae",
    "soondubu jigae": "soondubu jjigae"
}


# ========= LOADERS =========
@st.cache_data
def load_data():
    if not os.path.exists("review_sentiment.csv"):
        st.error("❌ 'review_sentiment.csv' not found.")
        return None
    try:
        df = pd.read_csv("review_sentiment.csv")
        df = df[df["sentiment"] == "positive"]
        df["menu"] = df["menu"].str.lower().str.replace("suundubu", "soondubu")
        df["menu"] = df["menu"].replace(menu_aliases)

        menu_stats = df.groupby("menu").agg(
            count=("menu", "count"),
            avg_sentiment=("compound_score", "mean")
        ).reset_index()

        scaler = MinMaxScaler()
        menu_stats[["count_norm", "sentiment_norm"]] = scaler.fit_transform(
            menu_stats[["count", "avg_sentiment"]]
        )
        menu_stats["score"] = (menu_stats["count_norm"] + menu_stats["sentiment_norm"]) / 2
        return menu_stats
    except Exception as e:
        st.error(f"❌ Error loading data: {e}")
        return None

menu_stats = load_data()
if menu_stats is None:
    st.stop()


@st.cache_resource
def load_bert_model_and_tokenizer():
    from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dropout, Dense
    from tensorflow.keras.models import Model

    tokenizer = BertTokenizer.from_pretrained("bert_tokenizer")
    bert = TFBertModel.from_pretrained("bert-base-uncased")
    bert.trainable = False

    bert_out_input = Input(shape=(MAX_LEN, 768), name="bert_output")
    x = Conv1D(128, kernel_size=5, activation='relu')(bert_out_input)
    x = MaxPooling1D(pool_size=2)(x)
    x = Bidirectional(LSTM(64, dropout=0.2, return_sequences=False))(x)
    x = Dropout(0.3)(x)
    output = Dense(1, activation='sigmoid')(x)

    sentiment_model = Model(inputs=bert_out_input, outputs=output)
    sentiment_model.load_weights("bert_cnn_blstm_resaved.keras")

    return tokenizer, bert, sentiment_model


def predict_sentiment(text, tokenizer, bert_model, sentiment_model, max_len=MAX_LEN):
    inputs = tokenizer(
        text,
        return_tensors="tf",
        truncation=True,
        padding="max_length",
        max_length=max_len
    )
    bert_output = bert_model(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    ).last_hidden_state

    preds = sentiment_model.predict(bert_output)
    return int(preds[0][0] > 0.5)


tokenizer, bert_model, bert_sentiment_model = load_bert_model_and_tokenizer()


# ========= UTILS =========
def correct_spelling(text):
    from textblob import TextBlob
    return str(TextBlob(text).correct())

def detect_category(text):
    text = text.lower()
    for keyword, category in keyword_aliases.items():
        if keyword in text:
            return category
    return None

def fuzzy_match_menu(text, menu_list):
    text = text.lower()
    for menu in menu_list:
        if all(word in text for word in menu.split()):
            return menu
    return None

def detect_negative_rule(text):
    negative_keywords = ["don't", "not", "dislike", "too", "hate", "worst", "bad"]
    return any(neg in text for neg in negative_keywords)

def is_category_only_input(text):
    words = text.lower().split()
    for word in words:
        if word not in keyword_aliases:
            return False
    return True


# ========= CHAT =========
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.title("👩‍🍳 GoStop Korean BBQ Menu Recommender")
st.markdown("Ask something like **'recommend me non-spicy food'** or **'how about odeng?'**")

with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_input("You:", placeholder="Type your request here...")
    submitted = st.form_submit_button("Send")

if submitted and user_input:
    raw_input = user_input.lower()
    corrected_input = user_input if len(user_input.split()) <= 2 else correct_spelling(user_input)
    corrected_lower = corrected_input.lower()

    matched_menu = fuzzy_match_menu(raw_input, menu_actual)
    category = detect_category(raw_input)
    is_category_input = is_category_only_input(corrected_lower)
    explicit_negative = detect_negative_rule(raw_input)
    is_negative = False
    sentiment_pred = "SKIPPED"

    if matched_menu and not explicit_negative:
        is_negative = False
    elif matched_menu and explicit_negative:
        is_negative = True
    elif matched_menu:
        sentiment_pred = predict_sentiment(corrected_input, tokenizer, bert_model, bert_sentiment_model)
        is_negative = sentiment_pred == 0
    elif category and not explicit_negative and is_category_input:
        is_negative = False
    elif category and explicit_negative:
        is_negative = True
    else:
        sentiment_pred = predict_sentiment(corrected_input, tokenizer, bert_model, bert_sentiment_model)
        is_negative = sentiment_pred == 0

    show_mood = any(word in raw_input for word in ["love", "like", "want", "enjoy"]) and not is_negative
    sentiment_note = "😊 Awesome! You're in a good mood! " if show_mood else "😕 No worries! I got you. " if is_negative else ""

    recommended = None
    if matched_menu:
        matched_menu = matched_menu.strip().lower()
        if is_negative:
            recommended = menu_stats[menu_stats["menu"] != matched_menu].sort_values("score", ascending=False).head(3)
            response = sentiment_note + f"Oops! You don't like <strong>{matched_menu.title()}</strong>? Try these instead:"
        elif matched_menu in menu_stats["menu"].values:
            row = menu_stats[menu_stats["menu"] == matched_menu].iloc[0]
            response = sentiment_note + f"🍽️ <strong>{matched_menu.title()}</strong> has <strong>{row['count']} reviews</strong> with average sentiment <strong>{row['avg_sentiment']:.2f}</strong>. Recommended! 🎉"
        elif matched_menu in menu_actual:
            response = f"🍽️ <strong>{matched_menu.title()}</strong> is on our menu! 🎉"
        else:
            recommended = menu_stats.sort_values("score", ascending=False).head(3)
            response = sentiment_note + "❌ Not sure about that menu. Here are our top 3 picks!"
    elif category and not matched_menu:
        matched = menu_categories.get(category, [])
        if is_negative:
            recommended = menu_stats[~menu_stats["menu"].isin(matched)].sort_values("score", ascending=False).head(3)
            response = f"🙅‍♂️ Avoiding <strong>{category.replace('_', ' ').title()}</strong>? Here are other ideas:"
        else:
            recommended = menu_stats[menu_stats["menu"].isin(matched)].sort_values("score", ascending=False).head(3)
            response = sentiment_note + f"🔥 You might like these <strong>{category.replace('_', ' ').title()}</strong> dishes:"
    else:
        recommended = menu_stats.sort_values("score", ascending=False).head(3)
        response = sentiment_note + "🤔 Couldn't find what you're looking for. Here's our top 3!"

    if recommended is not None:
        response += "<table><thead><tr><th>Rank</th><th>Menu</th><th>Sentiment</th><th>Reviews</th></tr></thead><tbody>"
        for idx, (_, row) in enumerate(recommended.iterrows(), 1):
            response += f"<tr style='text-align:center;'><td>{idx}</td><td>{row['menu'].title()}</td><td>{row['avg_sentiment']:.2f}</td><td>{int(row['count'])}</td></tr>"
        response += "</tbody></table>"

    status = "not recommended" if is_negative else ("recommended" if matched_menu or category else "fallback")
    log_row = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_input": user_input,
        "correction": corrected_input,
        "matched_menu": matched_menu if matched_menu else "none",
        "category": category if category else "none",
        "sentiment_polarity": "negative" if is_negative else ("positive" if sentiment_pred != "SKIPPED" else "none"),
        "status": status
    }

    file_exists = os.path.isfile(LOG_FILE)
    with open(LOG_FILE, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=log_row.keys())
        if not file_exists or os.path.getsize(LOG_FILE) == 0:
            writer.writeheader()
        writer.writerow(log_row)

    st.session_state.chat_history.append(("user", user_input))
    st.session_state.chat_history.append(("bot", response))

    for role, msg in st.session_state.chat_history:
        if role == "user":
            st.markdown(f"**You:** {msg}")
        else:
            st.markdown(msg, unsafe_allow_html=True)
