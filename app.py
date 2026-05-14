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
            'fatigue': 'Fatigue Score',
            'motivation': 'Motivation Score',
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
    if st.button('English'):
        st.session_state.lang = 'en'
        st.rerun()
with col2:
    if st.button('Espanol'):
        st.session_state.lang = 'es'
        st.rerun()

st.title(t('title'))
st.markdown(f"### {t('subtitle')}")

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('Emotion_Dataset_with_Self-Perceived_Performance12345.csv')
        return df
    except:
        import random
        emotions_list = ['Motivation', 'Anxiety', 'Fatigue', 'Frustration', 'Confidence', 'Resilience']
        text_by_emotion = {
            'Motivation': ["I'm ready to give 100 percent today", "Feeling unstoppable"],
            'Anxiety': ["Nights are sleepless before matchday", "Thinking too much before matches"],
            'Fatigue': ["Tired of training", "Body feels heavy"],
            'Frustration': ["Nothing is working", "Same mistakes again"],
            'Confidence': ["I can deliver when it matters", "Feeling confident"],
            'Resilience': ["I will bounce back", "Every challenge makes me better"]
        }
        data = []
        for i in range(500):
            emotion = random.choice(emotions_list)
            data.append({
                'text': random.choice(text_by_emotion[emotion]),
                'main_emotion': emotion,
                'fatigue_level': random.randint(1, 5),
                'motivation_level': random.randint(1, 5)
            })
        df = pd.DataFrame(data)
        return df

# Train models
@st.cache_resource
def train_models(df):
    emotion_encoder = LabelEncoder()
    y_emotion = emotion_encoder.fit_transform(df['main_emotion'])
    tfidf = TfidfVectorizer(max_features=2000)
    X_tfidf = tfidf.fit_transform(df['text'])
    emotion_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    emotion_model.fit(X_tfidf, y_emotion)
    fatigue_model = RandomForestRegressor(n_estimators=100, random_state=42)
    fatigue_model.fit(X_tfidf, df['fatigue_level'])
    # Use RandomForest for motivation too (instead of XGBoost)
    motivation_model = RandomForestRegressor(n_estimators=100, random_state=42)
    motivation_model.fit(X_tfidf, df['motivation_level'])
    return emotion_model, tfidf, emotion_encoder, fatigue_model, motivation_model

# Crisis detection
CRISIS_WORDS = {
    'suicide': ['suicide', 'kill myself', 'end my life', 'want to die', 'self harm'],
    'depression': ['depression', 'depressed', 'hopeless', 'worthless'],
    'anxiety': ['panic', 'terrified', 'scared to death']
}

def check_crisis(text):
    text_lower = text.lower()
    for crisis_type, keywords in CRISIS_WORDS.items():
        for keyword in keywords:
            if keyword in text_lower:
                return True, crisis_type
    return False, None

# Load models
with st.spinner("Loading models..."):
    df = load_data()
    emotion_model, tfidf, emotion_encoder, fatigue_model, motivation_model = train_models(df)
    st.success(f"Ready | {len(df)} samples | {df['main_emotion'].nunique()} emotions")

# Sidebar
with st.sidebar:
    st.markdown("### Dataset Stats")
    st.metric("Total Samples", len(df))
    st.metric("Emotion Classes", df['main_emotion'].nunique())
    st.markdown("### Emotion Distribution")
    for emo, count in df['main_emotion'].value_counts().head(5).items():
        st.write(f"- {emo}: {count}")

# Main layout
col_main, col_dash = st.columns([2, 1])

with col_main:
    st.markdown(f"### {t('enter_text')}")
    user_input = st.text_area("", height=150)
    
    if st.button(f"{t('predict')}", type="primary"):
        if user_input.strip():
            # Crisis check FIRST
            crisis, crisis_type = check_crisis(user_input)
            
            if crisis:
                st.error(f"🚨 {t('critical')} 🚨")
                st.error(f"Detected: {crisis_type.upper()}")
                st.markdown("**Immediate Actions:**")
                st.markdown("- Do not leave athlete alone")
                st.markdown("- Contact team psychologist immediately")
                st.markdown("- Emergency: 911 | Crisis Lifeline: 988")
            
            # ML predictions
            input_vec = tfidf.transform([user_input])
            emotion_pred = emotion_model.predict(input_vec)[0]
            emotion = emotion_encoder.inverse_transform([emotion_pred])[0]
            proba = emotion_model.predict_proba(input_vec)[0]
            confidence = max(proba) * 100
            fatigue_score = fatigue_model.predict(input_vec)[0]
            motivation_score = motivation_model.predict(input_vec)[0]
            
            # Display results
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                st.metric(t('emotion'), emotion, f"{confidence:.1f}%")
            with col_r2:
                st.metric(t('fatigue'), f"{fatigue_score:.1f}/5")
            with col_r3:
                st.metric(t('motivation'), f"{motivation_score:.1f}/5")
            
            # Interventions based on Word document
            st.markdown("---")
            if crisis:
                st.error(f"**{t('critical')} - Immediate psychiatric referral needed**")
            elif fatigue_score >= 4:
                st.warning(f"**{t('warning')}**")
                st.markdown("**Fatigue Intervention:**")
                st.markdown("- Reduce training load by 40% for 48 hours")
                st.markdown("- Ensure 8+ hours of sleep")
                st.markdown("- Schedule recovery session")
            elif emotion in ['Anxiety', 'Frustration'] and confidence > 70:
                st.warning(f"**{t('warning')}**")
                st.markdown("**Emotional Support Intervention:**")
                st.markdown("- Schedule one-on-one coach meeting")
                st.markdown("- Practice mindfulness exercises")
                st.markdown("- Adjust short-term goals")
            elif motivation_score <= 2:
                st.warning(f"**{t('warning')}**")
                st.markdown("**Motivation Intervention:**")
                st.markdown("- Set 3 small achievable goals")
                st.markdown("- Review past successes")
                st.markdown("- Team encouragement session")
            else:
                st.success(f"**{t('normal')}**")
                st.markdown("Continue standard monitoring and regular check-ins")
            
            # Save history
            if 'history' not in st.session_state:
                st.session_state.history = []
            st.session_state.history.insert(0, {
                'text': user_input[:80],
                'emotion': emotion,
                'fatigue': round(float(fatigue_score), 1),
                'motivation': round(float(motivation_score), 1),
                'time': datetime.now().strftime('%H:%M'),
                'crisis': crisis
            })
            st.session_state.history = st.session_state.history[:10]
        else:
            st.warning("Please enter some text")

with col_dash:
    st.markdown(f"### {t('stats')}")
    
    # Emotion distribution chart
    emotion_counts = df['main_emotion'].value_counts().head(6)
    fig = px.bar(x=emotion_counts.values, y=emotion_counts.index, orientation='h')
    fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f"### {t('history')}")
    if 'history' in st.session_state and st.session_state.history:
        for h in st.session_state.history[:5]:
            crisis_flag = "🚨 " if h.get('crisis', False) else ""
            st.markdown(f"**{h['time']}** - {crisis_flag}{h['emotion']} | F:{h['fatigue']} M:{h['motivation']}")
            st.caption(f"{h['text']}...")
            st.markdown("---")
    else:
        st.info("No predictions yet")

# Footer
st.markdown("---")
st.caption("EmotionSport AI | Emotion: Logistic Regression | Fatigue: Random Forest | Motivation: Random Forest | Crisis Detection: Keyword-based")
