import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import warnings
import os
import re
warnings.filterwarnings('ignore')

st.set_page_config(page_title="EmotionSport AI", page_icon="🧠", layout="wide")

st.title("🧠 EmotionSport AI")
st.markdown("### Real-time Emotion Classifier for Athletes")

# ============================================
# CRISIS KEYWORDS - HIGHEST PRIORITY
# ============================================
CRISIS_KEYWORDS = {
    'suicide': {
        'keywords': ['suicide', 'kill myself', 'end my life', 'want to die', 'better off dead', 'self harm', 'self-harm'],
        'level': 'CRITICAL',
        'action': '🚨 **IMMEDIATE CRISIS INTERVENTION REQUIRED** 🚨\n\nContact emergency services or suicide prevention hotline immediately.\n\n🏥 Emergency: 911\n📞 Suicide Prevention Lifeline: 988\n📱 Crisis Text Line: Text HOME to 741741'
    },
    'depression': {
        'keywords': ['depression', 'depressed', 'hopeless', 'worthless', 'no hope', 'empty inside', 'numb'],
        'level': 'HIGH',
        'action': '⚠️ **URGENT: Possible Depression Detected**\n\nImmediate psychological evaluation recommended. Do not leave athlete alone. Contact team psychologist.'
    },
    'self_harm': {
        'keywords': ['self harm', 'cut myself', 'hurt myself', 'self-injury', 'self injury', 'harming myself'],
        'level': 'CRITICAL',
        'action': '🚨 **CRITICAL: Self-Harm Indicators Detected** 🚨\n\nImmediate intervention required. Contact mental health professional now.'
    }
}

# Additional risk keywords
RISK_KEYWORDS = {
    'high': {
        'keywords': ['want to quit', 'give up', 'no reason to continue', 'cant go on', 'no point'],
        'action': '⚠️ HIGH RISK: Monitor closely. Schedule coach-athlete meeting today.'
    },
    'medium': {
        'keywords': ['frustrated', 'no chance to play', 'bench', 'dt no me da bola', 'don't care anymore', 'isolated', 'alone', 'no improvement', 'no me dan bola'],
        'action': '⚠️ MODERATE RISK: Coach check-in recommended within 24 hours.'
    }
}

def check_crisis(text):
    """Check for crisis keywords - HIGHEST PRIORITY"""
    text_lower = text.lower()
    
    # Check critical crisis first
    for crisis_type, data in CRISIS_KEYWORDS.items():
        for keyword in data['keywords']:
            if keyword in text_lower:
                return {
                    'detected': True,
                    'type': crisis_type,
                    'level': data['level'],
                    'message': data['action']
                }
    
    # Check other risks
    for risk_level, data in RISK_KEYWORDS.items():
        for keyword in data['keywords']:
            if keyword in text_lower:
                return {
                    'detected': True,
                    'type': 'risk',
                    'level': risk_level.upper(),
                    'message': data['action']
                }
    
    return {'detected': False}

# Show files for debugging
st.sidebar.markdown("### 📁 Files in server")
try:
    files = os.listdir('.')
    st.sidebar.write(files)
except:
    st.sidebar.write("Can't list files")

@st.cache_data
def load_data():
    # Use the CORRECT filename
    filename = "Emotion_Dataset_with_Self-Perceived_Performance12345.csv"
    
    if os.path.exists(filename):
        st.success(f"✅ Found file: {filename}")
        df = pd.read_csv(filename)
        
        st.sidebar.markdown("### 📊 Columns found")
        st.sidebar.write(list(df.columns))
        
        if 'text' not in df.columns:
            st.error(f"❌ 'text' column not found")
            return None
        if 'main_emotion' not in df.columns:
            st.error(f"❌ 'main_emotion' column not found")
            return None
        
        df['main_emotion'] = df['main_emotion'].fillna('Unknown')
        df['text'] = df['text'].fillna('').astype(str)
        df = df[df['text'].str.strip() != '']
        
        return df
    else:
        st.error(f"❌ File not found: {filename}")
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

# Warning about model limitations
st.sidebar.markdown("---")
st.sidebar.warning("⚠️ **Note:** The ML model has limitations. Crisis detection uses keyword matching as primary safety mechanism.")

# Main input
st.markdown("### 📝 Athlete Text Input")
user_input = st.text_area(
    "Enter text from interview, social media, or self-report:",
    height=150,
    placeholder="Example: 'I've been feeling really exhausted lately and can't seem to focus during training...'"
)

if st.button("🔮 Analyze & Predict", type="primary"):
    if not user_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        # STEP 1: CRISIS DETECTION (HIGHEST PRIORITY)
        crisis_result = check_crisis(user_input)
        
        if crisis_result['detected']:
            # Show crisis alert prominently
            if crisis_result['level'] == 'CRITICAL':
                st.error(f"""
                🚨🚨🚨 **CRITICAL ALERT** 🚨🚨🚨
                
                {crisis_result['message']}
                
                **Detected indicators:** {crisis_result['type']}
                """)
                
                # Additional resources
                with st.expander("📞 Emergency Resources", expanded=True):
                    st.markdown("""
                    **24/7 Crisis Support:**
                    - **988** - Suicide and Crisis Lifeline (call or text)
                    - **911** - Emergency Services
                    - **741741** - Crisis Text Line (text HOME)
                    - **1-800-273-8255** - National Suicide Prevention Lifeline
                    """)
            else:
                st.warning(f"""
                ⚠️ **CRISIS ALERT**
                
                {crisis_result['message']}
                """)
            
            # Still show ML prediction but with warning
            st.markdown("---")
            st.markdown("### 🤖 ML Model Analysis (with lower confidence due to crisis indicators)")
        
        # STEP 2: ML PREDICTION (but with disclaimer)
        try:
            input_vec = tfidf.transform([user_input])
            proba = model.predict_proba(input_vec)[0]
            pred = model.predict(input_vec)[0]
            emotion = label_encoder.inverse_transform([pred])[0]
            confidence = max(proba) * 100
            
            # Show ML results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("ML Detected Emotion", emotion, f"{confidence:.1f}% confidence")
            
            with col2:
                st.metric("Alert Status", "⚠️ FLAGGED" if crisis_result['detected'] else "✅ Normal")
            
            # Top 3 predictions
            top_3_idx = proba.argsort()[-3:][::-1]
            st.markdown("**ML Model Top Predictions:**")
            for idx in top_3_idx:
                emo = label_encoder.inverse_transform([idx])[0]
                st.progress(proba[idx], text=f"{emo}: {proba[idx]*100:.1f}%")
            
            # Warning about low confidence
            if confidence < 50:
                st.warning("⚠️ Low confidence prediction. Manual review recommended.")
            
            # Intervention based on emotion
            if not crisis_result['detected']:
                if emotion.lower() in ['fatigue', 'anxiety', 'frustration', 'insecurity']:
                    st.info("📋 **Recommendation:** Coach follow-up recommended within 48 hours.")
                elif emotion.lower() in ['motivation', 'resilience']:
                    st.success("✅ Positive indicators. Continue standard monitoring.")
                else:
                    st.info("📊 Routine monitoring continues.")
                    
        except Exception as e:
            st.error(f"ML Prediction error: {e}")

else:
    st.info("👆 Enter athlete text and click 'Analyze & Predict'")
