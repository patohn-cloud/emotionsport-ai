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

# Page config
st.set_page_config(page_title="EmotionSport AI", page_icon="🧠", layout="wide")

# Session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = None
if 'user_role' not in st.session_state:
    st.session_state.user_role = 'athlete'
if 'lang' not in st.session_state:
    st.session_state.lang = 'en'
if 'history' not in st.session_state:
    st.session_state.history = []

# Translations
def t(key):
    texts = {
        'en': {
            'title': 'EmotionSport AI',
            'subtitle': 'Professional Athlete Emotion and Wellness Monitoring',
            'login': 'Login',
            'logout': 'Logout',
            'username': 'Username',
            'password': 'Password',
            'welcome': 'Welcome',
            'dashboard': 'Dashboard',
            'emotion': 'Emotion',
            'fatigue': 'Fatigue',
            'motivation': 'Motivation',
            'risk_level': 'Risk Level',
            'enter_text': 'Enter athlete text',
            'predict': 'Analyze',
            'interventions': 'Interventions',
            'history': 'History',
            'stats': 'Stats',
            'help_title': 'Athlete Support Center - Complete Guide',
        },
        'es': {
            'title': 'EmotionSport AI',
            'subtitle': 'Monitoreo Profesional de Emociones y Bienestar',
            'login': 'Iniciar Sesion',
            'logout': 'Cerrar Sesion',
            'username': 'Usuario',
            'password': 'Contrasena',
            'welcome': 'Bienvenido',
            'dashboard': 'Panel',
            'emotion': 'Emocion',
            'fatigue': 'Fatiga',
            'motivation': 'Motivacion',
            'risk_level': 'Nivel de Riesgo',
            'enter_text': 'Ingrese texto del atleta',
            'predict': 'Analizar',
            'interventions': 'Intervenciones',
            'history': 'Historial',
            'stats': 'Estadisticas',
            'help_title': 'Centro de Apoyo - Guia Completa',
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

# Login
if not st.session_state.logged_in:
    with st.sidebar:
        st.markdown(f"### {t('login')}")
        username = st.text_input(t('username'))
        password = st.text_input(t('password'), type="password")
        if st.button(t('login'), type="primary"):
            if username and password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Enter username and password")
    st.info("Please login")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown(f"### {t('welcome')}, {st.session_state.username}!")
    st.markdown("---")
    
    if st.button("Contact Coach", use_container_width=True):
        st.info("Coach notification sent")
    if st.button("Contact Psychologist", use_container_width=True):
        st.info("Psychologist notified")
    if st.button("Emergency 988", use_container_width=True):
        st.error("Call 988 for crisis support - 24/7")
    
    st.markdown("---")
    if st.button("Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('Emotion_Dataset_with_Self-Perceived_Performance12345.csv')
    df['main_emotion'] = df['main_emotion'].fillna('Unknown')
    df['text'] = df['text'].fillna('').astype(str)
    df = df[df['text'].str.strip() != '']
    
    fatigue_map = {
        'fatigue': 5, 'anxiety': 4, 'frustration': 4, 'insecurity': 4,
        'motivation': 2, 'confidence': 2, 'resilience': 2, 'euphoria': 1
    }
    df['fatigue_level'] = df['main_emotion'].str.lower().map(fatigue_map).fillna(3)
    
    motivation_map = {
        'motivation': 5, 'euphoria': 5, 'confidence': 4, 'resilience': 4,
        'frustration': 2, 'anxiety': 2, 'fatigue': 1, 'insecurity': 1
    }
    df['motivation_level'] = df['main_emotion'].str.lower().map(motivation_map).fillna(3)
    
    return df

# Crisis detection
CRISIS_KEYWORDS = {
    'suicide': ['suicide', 'kill myself', 'end my life', 'want to die', 'better off dead'],
    'self_harm': ['self harm', 'cut myself', 'hurt myself', 'self injury'],
    'depression': ['depression', 'depressed', 'hopeless', 'worthless', 'empty inside'],
    'severe_anxiety': ['panic', 'terrified', 'scared to death', 'cant breathe']
}

LOW_MOTIVATION_WORDS = [
    'sin ganas', 'no quiero', 'me da igual', 'no me importa', 'para que',
    'abandonar', 'retirarme', 'dejarlo todo', 'no puedo mas', 'desmotivado',
    'sin energia', 'no veo mejoria', 'self harm', 'no tiene sentido'
]

def check_crisis(text):
    text_lower = text.lower()
    for crisis_type, words in CRISIS_KEYWORDS.items():
        for word in words:
            if word in text_lower:
                return True, crisis_type
    return False, None

def get_motivation_penalty(text):
    text_lower = text.lower()
    for word in LOW_MOTIVATION_WORDS:
        if word in text_lower:
            return -3  # Reduce motivation by 3 points
    return 0

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
    motivation_model = RandomForestRegressor(n_estimators=100, random_state=42)
    motivation_model.fit(X_tfidf, df['motivation_level'])
    return emotion_model, tfidf, emotion_encoder, fatigue_model, motivation_model

with st.spinner("Loading models..."):
    df = load_data()
    emotion_model, tfidf, emotion_encoder, fatigue_model, motivation_model = train_models(df)

with st.sidebar:
    st.markdown("---")
    st.metric("Total Samples", len(df))
    st.metric("Emotion Classes", df['main_emotion'].nunique())

# Main layout
col_main, col_dash = st.columns([2, 1])

with col_main:
    st.markdown(f"### {t('enter_text')}")
    user_input = st.text_area("", height=120)
    
    if st.button(f"{t('predict')}", type="primary"):
        if user_input.strip():
            # Crisis check
            crisis, crisis_type = check_crisis(user_input)
            motivation_penalty = get_motivation_penalty(user_input)
            
            # Predictions
            input_vec = tfidf.transform([user_input])
            emotion_pred = emotion_model.predict(input_vec)[0]
            emotion = emotion_encoder.inverse_transform([emotion_pred])[0]
            proba = emotion_model.predict_proba(input_vec)[0]
            confidence = max(proba) * 100
            fatigue_score = fatigue_model.predict(input_vec)[0]
            motivation_score = motivation_model.predict(input_vec)[0]
            
            # Apply penalty
            motivation_score = max(1.0, motivation_score + motivation_penalty)
            
            # Override if crisis
            if crisis:
                emotion = 'CRISIS'
                fatigue_score = 5.0
                motivation_score = 1.0
                confidence = 100.0
            
            # Crisis alert
            if crisis:
                st.error(f"""
                🚨🚨🚨 CRITICAL CRISIS INTERVENTION 🚨🚨🚨
                
                Detected: {crisis_type.upper()}
                
                IMMEDIATE ACTIONS:
                1. Do NOT leave athlete alone
                2. Call 988 (Suicide Crisis Lifeline)
                3. Contact team psychologist
                
                Emergency: 911
                """)
            
            # Metrics
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                st.metric(t('emotion'), emotion, f"{confidence:.0f}%")
            with col_r2:
                st.metric(t('fatigue'), f"{fatigue_score:.1f}/5")
            with col_r3:
                st.metric(t('motivation'), f"{motivation_score:.1f}/5")
            
            # Risk level
            if crisis or fatigue_score >= 4.5 or motivation_score <= 1.5:
                risk = "CRITICAL RISK"
                st.error(f"**Risk Level:** {risk}")
            elif fatigue_score >= 3.5 or motivation_score <= 2.5:
                risk = "MODERATE RISK"
                st.warning(f"**Risk Level:** {risk}")
            else:
                risk = "LOW RISK"
                st.success(f"**Risk Level:** {risk}")
            
            # Interventions
            st.markdown(f"### {t('interventions')}")
            
            if crisis:
                st.markdown("""
                **CRITICAL INTERVENTION:**
                
                Immediate:
                - Ensure physical safety
                - Stay with the person
                - Call 988 now
                
                Resources:
                - 988 Suicide and Crisis Lifeline
                - Crisis Text Line: Text HOME to 741741
                - Emergency: 911
                """)
            elif fatigue_score >= 4:
                st.markdown("""
                **FATIGUE RECOVERY PROTOCOL:**
                
                - Mandatory rest day today
                - Reduce training load by 50% for 48 hours
                - Aim for 9-10 hours of sleep
                - Hydrate with electrolytes
                - Ice bath or contrast shower
                """)
            elif emotion == 'Anxiety' and confidence > 60:
                st.markdown("""
                **ANXIETY MANAGEMENT:**
                
                Breathing Technique (4-7-8):
                - Inhale 4 seconds
                - Hold 7 seconds
                - Exhale 8 seconds
                - Repeat 5 times
                
                Grounding Technique (5-4-3-2-1):
                - 5 things you see
                - 4 things you touch
                - 3 things you hear
                - 2 things you smell
                - 1 thing you taste
                """)
            elif emotion == 'Frustration' and confidence > 60:
                st.markdown("""
                **FRUSTRATION MANAGEMENT:**
                
                In the moment:
                - Deep breath (4 in, 4 out)
                - Shake out arms and legs
                - Say "Reset" or "Next"
                - Focus on next play only
                
                After session:
                - Journal what triggered it
                - Identify what you can control
                - Set realistic expectations
                """)
            elif motivation_score <= 2.5:
                st.markdown("""
                **MOTIVATION BOOSTER:**
                
                5-Minute Actions:
                1. Watch 2 minutes of your best highlights
                2. Power pose for 2 minutes
                3. Write 3 things going well
                4. Listen to pump-up music
                5. Text a supportive teammate
                
                Set one small goal for today. Start small. Build momentum.
                """)
            else:
                st.markdown("""
                **NORMAL STATUS - OPTIMIZATION MODE:**
                
                - Continue current training load
                - Add one small challenge this week
                - Track your wins daily
                - Maintain recovery routines
                """)
            
            # Save history
            st.session_state.history.insert(0, {
                'text': user_input[:80],
                'emotion': emotion,
                'fatigue': round(float(fatigue_score), 1),
                'motivation': round(float(motivation_score), 1),
                'crisis': crisis,
                'risk': risk,
                'time': datetime.now().strftime('%H:%M')
            })
            st.session_state.history = st.session_state.history[:10]
        else:
            st.warning("Please enter some text")

# Right column
with col_dash:
    st.markdown(f"### {t('dashboard')}")
    
    if st.session_state.history:
        last = st.session_state.history[0]
        st.markdown(f"""
        **Current Status:**
        - Emotion: {last['emotion']}
        - Fatigue: {last['fatigue']}/5
        - Motivation: {last['motivation']}/5
        - Risk: {last['risk']}
        """)
        
        if last['motivation'] <= 2:
            st.progress(last['motivation']/5, text="Motivation Critical")
        else:
            st.progress(last['motivation']/5, text="Motivation Level")
    
    # Chart
    emotion_counts = df['main_emotion'].value_counts().head(6)
    fig = px.bar(x=emotion_counts.values, y=emotion_counts.index, orientation='h')
    fig.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)
    
    # History
    st.markdown(f"### {t('history')}")
    if st.session_state.history:
        for h in st.session_state.history[:5]:
            crisis_icon = "🚨 " if h.get('crisis', False) else ""
            st.markdown(f"**{h['time']}** - {crisis_icon}{h['emotion']}")
            st.markdown(f"Fatigue: {h['fatigue']}/5 | Motivation: {h['motivation']}/5")
            st.caption(f"{h['text']}...")
            st.markdown("---")
    else:
        st.info("No predictions yet")

# Help section
with st.expander(f"📚 {t('help_title')}", expanded=False):
    tab1, tab2, tab3, tab4 = st.tabs([
        "Mental Health Crisis", "Anxiety Management", 
        "Frustration & Motivation", "Performance & Recovery"
    ])
    
    with tab1:
        st.markdown("""
        **MENTAL HEALTH CRISIS - WHAT TO DO**
        
        **Call 988** - Suicide and Crisis Lifeline (24/7)
        
        **Warning Signs:**
        - Talking about wanting to die
        - Feeling hopeless or trapped
        - Withdrawing from others
        - Giving away possessions
        
        **What to do:**
        1. Ask directly about suicidal thoughts
        2. Keep them safe - remove dangerous items
        3. Listen without judgment
        4. Help them call 988
        5. Follow up within 48 hours
        
        **Resources:**
        - 988: Suicide and Crisis Lifeline
        - Text HOME to 741741: Crisis Text Line
        - 911: Emergency Services
        """)
    
    with tab2:
        st.markdown("""
        **ANXIETY MANAGEMENT FOR ATHLETES**
        
        **4-7-8 Breathing:**
        - Inhale for 4 seconds
        - Hold for 7 seconds
        - Exhale for 8 seconds
        - Repeat 5 times
        
        **5-4-3-2-1 Grounding:**
        - 5 things you see
        - 4 things you touch
        - 3 things you hear
        - 2 things you smell
        - 1 thing you taste
        
        **Pre-Competition Routine:**
        - Night before: Lay out equipment
        - Morning: Normal breakfast
        - 1 hour before: Light warm-up
        - 10 minutes before: Cue words: "Calm - Focus"
        
        **See a sports psychologist if anxiety affects sleep or performance**
        """)
    
    with tab3:
        st.markdown("""
        **HANDLING FRUSTRATION & LOW MOTIVATION**
        
        **In the moment:**
        - Deep breath, shake it off
        - Say "Next" or "Reset"
        - Focus on the next play only
        
        **Boosting motivation (5 minutes):**
        - Watch 2 minutes of your best moments
        - Power pose (hands on hips, chest out)
        - Write 3 things going well
        - Listen to pump-up music
        
        **Set SMART goals:**
        - Specific: What exactly?
        - Measurable: How to track?
        - Achievable: Realistic?
        - Relevant: Why important?
        - Time-bound: When by?
        
        **Remember: Motivation follows action - start small**
        """)
    
    with tab4:
        st.markdown("""
        **PERFORMANCE OPTIMIZATION & RECOVERY**
        
        **Sleep for athletes:**
        - Aim for 8-10 hours
        - No screens 1 hour before bed
        - Cool, dark room
        - Consistent schedule
        
        **Nutrition:**
        - Eat within 30 minutes after training
        - Protein + carbs for recovery
        - Hydrate: 2-3 liters daily
        
        **Training:**
        - Progressive overload: Increase 5-10% weekly
        - Rest 2-3 minutes between heavy sets
        - Don't skip warm-up and cool-down
        
        **Signs of overtraining:**
        - Persistent fatigue
        - Poor sleep
        - Irritability
        - Decreased performance
        - Take 2-3 days of light work if you see these signs
        """)

# Footer
st.markdown("---")
st.caption("EmotionSport AI - Professional Athlete Monitoring System")
