import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="EmotionSport AI", page_icon="🧠", layout="wide")

st.title("🧠 EmotionSport AI")
st.markdown("### Real-time Emotion Classifier for Athletes")

@st.cache_data
def load_data():
    df = pd.read_csv("Ultimate_Emotion_Dataset_With_All_Features.csv")
    df['main_emotion'] = df['main_emotion'].fillna('Unknown')
    df['text'] = df['text'].fillna('').astype(str)
    df = df[df['text'].str.strip() != '']
    return df

@st.cache_resource
def train_model(df):
    X = df['text'].values
    y = df['main_emotion'].values
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    tfidf = TfidfVectorizer(max_features=2000)
    X_tfidf = tfidf.fit_transform(X)
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    model.fit(X_tfidf, y_encoded)
    return model, tfidf, label_encoder

with st.spinner("Loading model..."):
    df = load_data()
    model, tfidf, label_encoder = train_model(df)
    st.success(f"✅ Ready! {len(df)} samples, {df['main_emotion'].nunique()} emotions")

st.sidebar.markdown("### 📊 Stats")
st.sidebar.metric("Total Samples", len(df))
st.sidebar.metric("Emotion Classes", df['main_emotion'].nunique())

user_input = st.text_area("📝 Enter athlete text (interview, social media, self-report):", height=150)

if st.button("🔮 Predict Emotion", type="primary"):
    if user_input.strip():
        input_vec = tfidf.transform([user_input])
        pred = model.predict(input_vec)[0]
        emotion = label_encoder.inverse_transform([pred])[0]
        proba = model.predict_proba(input_vec)[0]
        confidence = max(proba) * 100
        
        st.success(f"**Detected Emotion:** {emotion} ({confidence:.1f}% confidence)")
        
        if emotion in ['Mental Fatigue', 'Frustration', 'Pre-Match Anxiety']:
            st.error("⚠️ **URGENT INTERVENTION NEEDED** - Contact coaching/psychology staff immediately")
        else:
            st.info("✅ No urgent intervention needed - Continue monitoring")
    else:
        st.warning("Please enter some text")