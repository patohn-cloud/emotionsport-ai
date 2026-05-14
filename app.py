import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page setup
st.set_page_config(page_title="EmotionSport AI", page_icon="🧠", layout="wide")

# Language
if 'lang' not in st.session_state:
    st.session_state.lang = 'en'

def t(key):
    texts = {
        'en': {
            'title': 'EmotionSport AI',
            'subtitle': 'Professional Athlete Emotion & Wellness Monitoring',
            'emotion': 'Detected Emotion',
            'fatigue': 'Fatigue Level',
            'motivation': 'Motivation Level',
            'critical': 'CRITICAL - Immediate Intervention Required',
            'warning': 'WARNING - Coach Follow-up Recommended',
            'normal': 'Normal - Continue Monitoring',
            'enter_text': 'Enter athlete text:',
            'predict': 'Analyze Emotion & Risk',
            'stats': 'Dashboard',
            'history': 'Recent History'
        },
        'es': {
            'title': 'EmotionSport AI',
            'subtitle': 'Monitoreo Profesional de Emociones',
            'emotion': 'Emocion Detectada',
            'fatigue': 'Nivel de Fatiga',
            'motivation': 'Nivel de Motivacion',
            'critical': 'CRITICO - Intervencion Inmediata',
            'warning': 'ADVERTENCIA - Seguimiento Recomendado',
            'normal': 'Normal - Continuar Monitoreo',
            'enter_text': 'Ingrese texto del atleta:',
            'predict': 'Analizar Emocion y Riesgo',
            'stats': 'Panel de Control',
            'history': 'Historial Reciente'
        }
    }
    return texts[st.session_state.lang][key]

# Language selector
col1, col2 = st.columns(2)
with col1:
    if st.button('🇬🇧 English'):
        st.session_state.lang = 'en'
        st.rerun()
with col2:
    if st.button('🇪🇸 Español'):
        st.session_state.lang = 'es'
        st.rerun()

st.title(t('title'))
st.markdown(f"### {t('subtitle')}")

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('Emotion_Dataset_with_Self-Perceived_Performance12345.csv')
    
    # Show columns in sidebar for debugging
    st.sidebar.markdown("### 📋 CSV Columns")
    st.sidebar.write(list(df.columns))
    
    # Clean data
    df['main_emotion'] = df['main_emotion'].fillna('Unknown')
    df['text'] = df['text'].fillna('').astype(str)
    df = df[df['text'].str.strip() != '']
    
    # Create fatigue_level from emotion_level if not exists
    if 'fatigue_level' not in df.columns:
        # Map emotions to fatigue levels
        fatigue_map = {
            'fatigue': 5,
            'anxiety': 4,
            'frustration': 4,
            'insecurity': 3,
            'motivation': 3,
            'confidence': 2,
            'resilience': 2,
            'euphoria': 1
        }
        df['fatigue_level'] = df['main_emotion'].str.lower().map(fatigue_map).fillna(3)
    
    # Create motivation_level from emotion_level if not exists
    if 'motivation_level' not in df.columns:
        # Map emotions to motivation levels
        motivation_map = {
            'motivation': 5,
            'euphoria': 5,
            'confidence': 4,
            'resilience': 4,
            'frustration': 2,
            'anxiety': 2,
            'fatigue': 2,
            'insecurity': 2
        }
        df['motivation_level'] = df['main_emotion'].str.lower().map(motivation_map).fillna(3)
    
    return df

# Train models
@st.cache_resource
def train_models(df):
    # Emotion classifier
    emotion_encoder = LabelEncoder()
    y_emotion = emotion_encoder.fit_transform(df['main_emotion'])
    
    tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(df['text'])
    
    emotion_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    emotion_model.fit(X_tfidf, y_emotion)
    
    # Fatigue predictor
    fatigue_model = RandomForestRegressor(n_estimators=100, random_state=42)
    fatigue_model.fit(X_tfidf, df['fatigue_level'])
    
    # Motivation predictor
    motivation_model = RandomForestRegressor(n_estimators=100, random_state=42)
    motivation_model.fit(X_tfidf, df['motivation_level'])
    
    return emotion_model, tfidf, emotion_encoder, fatigue_model, motivation_model, df

# Crisis detection (high priority)
CRISIS_WORDS = {
    'suicide': ['suicide', 'kill myself', 'end my life', 'want to die', 'self harm', 'self-harm'],
    'depression': ['depression', 'depressed', 'hopeless', 'worthless', 'numb', 'empty'],
    'severe_anxiety': ['panic', 'terrified', 'scared to death', 'cant breathe'],
    'self_harm': ['cut myself', 'hurt myself', 'injury myself']
}

def check_crisis(text):
    text_lower = text.lower()
    for crisis_type, keywords in CRISIS_WORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                return True, crisis_type
    return False, None

# Load everything
with st.spinner("Loading models..."):
    df = load_data()
    emotion_model, tfidf, emotion_encoder, fatigue_model, motivation_model, df = train_models(df)
    st.success(f"✅ Ready | {len(df)} samples | {df['main_emotion'].nunique()} emotions")

# Sidebar stats
with st.sidebar:
    st.markdown("### 📊 Dataset Stats")
    st.metric("Total Samples", len(df))
    st.metric("Emotion Classes", df['main_emotion'].nunique())
    
    st.markdown("### 🎭 Emotion Distribution")
    for emo, count in df['main_emotion'].value_counts().head(8).items():
        st.progress(count/len(df), text=f"{emo}: {count}")
    
    st.markdown("### ⚠️ Crisis Keywords Monitored")
    st.caption("suicide, self harm, depression, hopeless, kill myself, panic")

# Main layout
col_main, col_dash = st.columns([2, 1])

with col_main:
    st.markdown(f"### 📝 {t('enter_text')}")
    user_input = st.text_area("", height=150, placeholder="Example: 'I've been feeling really exhausted and can't focus...'")
    
    if st.button(f"🔮 {t('predict')}", type="primary"):
        if user_input.strip():
            # STEP 1: CRISIS CHECK (HIGHEST PRIORITY)
            crisis, crisis_type = check_crisis(user_input)
            
            if crisis:
                st.error(f"""
                🚨🚨🚨 **{t('critical')}** 🚨🚨🚨
                
                **Detected: {crisis_type.upper()}**
                
                **Immediate Actions:**
                - 🛑 Do NOT leave athlete alone
                - 📞 Contact team psychologist NOW
                - 🏥 Emergency: 911
                - 📱 Crisis Lifeline: 988 (24/7)
                """)
            
            # STEP 2: ML PREDICTIONS
            input_vec = tfidf.transform([user_input])
            
            # Emotion prediction
            emotion_pred = emotion_model.predict(input_vec)[0]
            emotion = emotion_encoder.inverse_transform([emotion_pred])[0]
            proba = emotion_model.predict_proba(input_vec)[0]
            confidence = max(proba) * 100
            
            # Fatigue prediction
            fatigue_score = fatigue_model.predict(input_vec)[0]
            
            # Motivation prediction
            motivation_score = motivation_model.predict(input_vec)[0]
            
            # Display results
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                st.metric(t('emotion'), emotion, f"{confidence:.1f}%")
            with col_r2:
                st.metric(t('fatigue'), f"{fatigue_score:.1f}/5")
            with col_r3:
                st.metric(t('motivation'), f"{motivation_score:.1f}/5")
            
            # STEP 3: INTERVENTIONS based on Word document
            st.markdown("---")
            st.markdown("### 📋 Recommended Intervention")
            
            if crisis:
                st.error("**CRITICAL: Psychiatric emergency protocol activated**")
                st.markdown("""
                1. 🚨 Ensure athlete safety immediately
                2. 📞 Emergency services if needed  
                3. 🧠 Contact team psychologist
                4. 🛑 Remove from all activities
                """)
            elif fatigue_score >= 4:
                st.warning("**⚠️ High Fatigue Detected**")
                st.markdown("""
                **Intervention:**
                - 📉 Reduce training load by 40%
                - 😴 Ensure 8+ hours sleep
                - 📞 Schedule recovery session
                - 📊 Monitor for 48 hours
                """)
            elif emotion in ['Anxiety', 'Frustration'] and confidence > 70:
                st.warning("**⚠️ Emotional Distress Detected**")
                st.markdown("""
                **Intervention:**
                - 🗣️ One-on-one coach meeting
                - 🧘 Mindfulness exercises
                - 🎯 Adjust short-term goals
                - 📝 Daily check-ins
                """)
            elif motivation_score <= 2:
                st.warning("**⚠️ Low Motivation Detected**")
                st.markdown("""
                **Intervention:**
                - 🏆 Set 3 small achievable goals
                - 📝 Review past successes
                - 👥 Team encouragement session
                - 🎯 Focus on one skill at a time
                """)
            else:
                st.success("**✅ Normal Status**")
                st.markdown("Continue standard monitoring and regular wellness check-ins")
            
            # Save to history
            if 'history' not in st.session_state:
                st.session_state.history = []
            st.session_state.history.insert(0, {
                'text': user_input[:80],
                'emotion': emotion,
                'fatigue': round(float(fatigue_score), 1),
                'motivation': round(float(motivation_score), 1),
                'confidence': round(confidence, 1),
                'crisis': crisis,
                'time': datetime.now().strftime('%H:%M')
            })
            st.session_state.history = st.session_state.history[:10]
            
        else:
            st.warning("Please enter some text to analyze")

with col_dash:
    st.markdown(f"### 📈 {t('stats')}")
    
    # Show emotion distribution chart
    emotion_counts = df['main_emotion'].value_counts().head(8)
    fig = px.bar(x=emotion_counts.values, y=emotion_counts.index, orientation='h', 
                 title="Training Data Distribution",
                 labels={'x': 'Count', 'y': 'Emotion'})
    fig.update_layout(height=350, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f"### 📋 {t('history')}")
    if 'history' in st.session_state and st.session_state.history:
        for h in st.session_state.history[:5]:
            crisis_icon = "🚨 " if h.get('crisis', False) else ""
            st.markdown(f"**{h['time']}** - {crisis_icon}{h['emotion']} ({h['confidence']}%)")
            st.markdown(f"Fatigue: {h['fatigue']}/5 | Motivation: {h['motivation']}/5")
            st.caption(f"📝 {h['text']}...")
            st.markdown("---")
    else:
        st.info("No predictions yet. Enter text and click Analyze.")

# Footer
st.markdown("---")
st.caption("EmotionSport AI | Emotion: Logistic Regression | Fatigue: Random Forest | Motivation: Random Forest | Crisis Detection: Keyword Matching")
