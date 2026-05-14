import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import warnings
import os
warnings.filterwarnings('ignore')

st.set_page_config(page_title="EmotionSport AI", page_icon="🧠", layout="wide")

st.title("🧠 EmotionSport AI")
st.markdown("### Real-time Emotion Classifier for Athletes")

# Show files for debugging
st.sidebar.markdown("### 📁 Files in server")
try:
    files = os.listdir('.')
    st.sidebar.write(files)
except:
    st.sidebar.write("Can't list files")

@st.cache_data
def load_data():
    # Use the CORRECT filename from your CSV
    filename = "Emotion_Dataset_with_Self-Perceived_Performance12345.csv"
    
    if os.path.exists(filename):
        st.success(f"✅ Found file: {filename}")
        df = pd.read_csv(filename)
        
        # Display column names for debugging
        st.sidebar.markdown("### 📊 Columns found")
        st.sidebar.write(list(df.columns))
        
        # Check required columns
        if 'text' not in df.columns:
            st.error(f"❌ 'text' column not found. Columns: {list(df.columns)}")
            return None
        if 'main_emotion' not in df.columns:
            st.error(f"❌ 'main_emotion' column not found. Columns: {list(df.columns)}")
            return None
        
        # Clean data
        df['main_emotion'] = df['main_emotion'].fillna('Unknown')
        df['text'] = df['text'].fillna('').astype(str)
        df = df[df['text'].str.strip() != '']
        
        return df
    else:
        st.error(f"❌ File not found: {filename}")
        st.markdown(f"Available files: {files}")
        return None

@st.cache_resource
def train_model(df):
    X = df['text'].values
    y = df['main_emotion'].values
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(X)
    
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_tfidf, y_encoded)
    
    return model, tfidf, label_encoder

# Load data
df = load_data()

if df is None:
    st.stop()

# Train model
with st.spinner("Training model..."):
    model, tfidf, label_encoder = train_model(df)
    st.success(f"✅ Ready! {len(df)} samples, {df['main_emotion'].nunique()} emotions")

# Sidebar stats
st.sidebar.markdown("### 📊 Dataset Stats")
st.sidebar.metric("Total Samples", len(df))
st.sidebar.metric("Emotion Classes", df['main_emotion'].nunique())

st.sidebar.markdown("### 🎭 Top Emotions")
emotion_counts = df['main_emotion'].value_counts().head(10)
for emo, count in emotion_counts.items():
    st.sidebar.write(f"- {emo}: {count}")

# Main input
user_input = st.text_area("📝 Enter athlete text (interview, social media, self-report):", height=150)

if st.button("🔮 Predict Emotion", type="primary"):
    if user_input.strip():
        input_vec = tfidf.transform([user_input])
        pred = model.predict(input_vec)[0]
        emotion = label_encoder.inverse_transform([pred])[0]
        proba = model.predict_proba(input_vec)[0]
        confidence = max(proba) * 100
        
        st.success(f"**Detected Emotion:** {emotion} ({confidence:.1f}% confidence)")
        
        # Show top 3 predictions
        top_3_idx = proba.argsort()[-3:][::-1]
        st.markdown("**Top predictions:**")
        for idx in top_3_idx:
            emo = label_encoder.inverse_transform([idx])[0]
            st.progress(proba[idx], text=f"{emo}: {proba[idx]*100:.1f}%")
        
        # Interventions based on emotion
        critical_emotions = ['fatigue', 'frustration', 'anxiety', 'insecurity']
        if emotion.lower() in critical_emotions:
            st.error("⚠️ **URGENT INTERVENTION NEEDED**")
            st.markdown("""
            **Recommended Actions:**
            - 🛑 Consider rest or reduced training load
            - 📞 Schedule coach or psychologist check-in
            - 📝 Document emotional state daily
            - 💬 Open communication encouraged
            """)
        else:
            st.info("✅ No urgent intervention needed - Continue monitoring")
    else:
        st.warning("Please enter some text")
