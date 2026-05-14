import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import plotly.express as px
import plotly.graph_objects as go
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
            'subtitle': 'Professional Athlete Emotion & Wellness Monitoring',
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
            'enter_text': 'Enter athlete text:',
            'predict': 'Analyze',
            'interventions': 'Interventions',
            'history': 'History',
            'stats': 'Stats',
            'help_title': 'Athlete Support Center',
            'help_crisis': 'Mental Health Crisis',
            'help_anxiety': 'Managing Anxiety',
            'help_gym': 'Gym Improvement',
            'help_performance': 'Performance Optimization',
            'help_sleep': 'Sleep & Recovery',
            'help_nutrition': 'Sports Nutrition',
        },
        'es': {
            'title': 'EmotionSport AI',
            'subtitle': 'Monitoreo de Emociones para Atletas',
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
            'enter_text': 'Ingrese texto del atleta:',
            'predict': 'Analizar',
            'interventions': 'Intervenciones',
            'history': 'Historial',
            'stats': 'Estadisticas',
            'help_title': 'Centro de Apoyo',
            'help_crisis': 'Crisis de Salud Mental',
            'help_anxiety': 'Manejar Ansiedad',
            'help_gym': 'Mejorar en Gimnasio',
            'help_performance': 'Optimizar Rendimiento',
            'help_sleep': 'Sueno y Recuperacion',
            'help_nutrition': 'Nutricion Deportiva',
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
    if st.button('🇪🇸 Espanol'):
        st.session_state.lang = 'es'
        st.rerun()

st.title(t('title'))
st.markdown(f"### {t('subtitle')}")

# Login
if not st.session_state.logged_in:
    with st.sidebar:
        st.markdown(f"### 🔐 {t('login')}")
        username = st.text_input(t('username'))
        password = st.text_input(t('password'), type="password")
        if st.button(t('login'), type="primary"):
            if username and password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Enter username and password")
    st.info("👈 Please login")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown(f"### 👋 {t('welcome')}, {st.session_state.username}!")
    st.markdown("---")
    if st.button("📞 Contact Coach", use_container_width=True):
        st.info("Coach notified (demo)")
    if st.button("🧠 Contact Psychologist", use_container_width=True):
        st.info("Psychologist notified (demo)")
    if st.button("🚨 Emergency (988)", use_container_width=True):
        st.error("Call 988 for crisis support")
    st.markdown("---")
    if st.button("🚪 " + t('logout'), use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()

# Load data and models
@st.cache_data
def load_data():
    df = pd.read_csv('Emotion_Dataset_with_Self-Perceived_Performance12345.csv')
    df['main_emotion'] = df['main_emotion'].fillna('Unknown')
    df['text'] = df['text'].fillna('').astype(str)
    df = df[df['text'].str.strip() != '']
    
    fatigue_map = {'fatigue': 5, 'anxiety': 4, 'frustration': 4, 'insecurity': 3,
                   'motivation': 3, 'confidence': 2, 'resilience': 2, 'euphoria': 1}
    df['fatigue_level'] = df['main_emotion'].str.lower().map(fatigue_map).fillna(3)
    
    motivation_map = {'motivation': 5, 'euphoria': 5, 'confidence': 4, 'resilience': 4,
                      'frustration': 2, 'anxiety': 2, 'fatigue': 2, 'insecurity': 2}
    df['motivation_level'] = df['main_emotion'].str.lower().map(motivation_map).fillna(3)
    return df

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

def check_crisis(text):
    crisis_words = ['suicide', 'kill myself', 'self harm', 'depression', 'hopeless']
    text_lower = text.lower()
    for word in crisis_words:
        if word in text_lower:
            return True
    return False

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
    st.markdown(f"### 📝 {t('enter_text')}")
    user_input = st.text_area("", height=120)
    
    if st.button(f"🔮 {t('predict')}", type="primary"):
        if user_input.strip():
            # Crisis check
            crisis = check_crisis(user_input)
            
            if crisis:
                st.error("""
                🚨🚨🚨 **CRISIS INTERVENTION PROTOCOL** 🚨🚨🚨
                
                **Immediate Actions:**
                - Do NOT leave athlete alone
                - Call 988 (Suicide Crisis Lifeline)
                - Contact team psychologist
                
                **Emergency: 911**
                """)
            
            # Predictions
            input_vec = tfidf.transform([user_input])
            emotion_pred = emotion_model.predict(input_vec)[0]
            emotion = emotion_encoder.inverse_transform([emotion_pred])[0]
            proba = emotion_model.predict_proba(input_vec)[0]
            confidence = max(proba) * 100
            fatigue_score = fatigue_model.predict(input_vec)[0]
            motivation_score = motivation_model.predict(input_vec)[0]
            
            # Display metrics
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                st.metric(t('emotion'), emotion, f"{confidence:.0f}%")
            with col_r2:
                st.metric(t('fatigue'), f"{fatigue_score:.1f}/5")
            with col_r3:
                st.metric(t('motivation'), f"{motivation_score:.1f}/5")
            
            # Interventions
            st.markdown(f"### 📋 {t('interventions')}")
            
            if crisis:
                st.markdown("""
                **CRISIS RESPONSE:**
                1. Ensure immediate safety
                2. Call 988 or 911
                3. Remove from all activities
                4. Contact family if appropriate
                """)
            elif fatigue_score >= 4:
                st.markdown("""
                **FATIGUE INTERVENTION:**
                - Mandatory rest day
                - Reduce training load by 50%
                - Ensure 8-10 hours sleep
                - Hydrate and recover
                """)
            elif emotion in ['Anxiety', 'Frustration'] and confidence > 70:
                st.markdown("""
                **ANXIETY/FRUSTRATION MANAGEMENT:**
                - Box breathing: 4-7-8 technique
                - Visualization of success
                - Coach meeting scheduled
                - Mindfulness exercises
                """)
            elif motivation_score <= 2:
                st.markdown("""
                **MOTIVATION BOOSTER:**
                - Set 3 small daily goals
                - Watch highlight videos
                - Talk to teammates
                - Focus on one skill at a time
                """)
            else:
                st.markdown("✅ **Normal status - continue monitoring**")
            
            # Save history
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
    st.markdown(f"### 📈 {t('dashboard')}")
    
    if st.session_state.history:
        last = st.session_state.history[0]
        st.markdown(f"""
        **Current Status:**
        - Emotion: {last['emotion']}
        - Fatigue: {last['fatigue']}/5
        - Motivation: {last['motivation']}/5
        """)
    
    # Chart
    emotion_counts = df['main_emotion'].value_counts().head(6)
    fig = px.bar(x=emotion_counts.values, y=emotion_counts.index, orientation='h')
    fig.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown(f"### 📋 {t('history')}")
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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        t('help_crisis'), t('help_anxiety'), t('help_gym'),
        t('help_performance'), t('help_sleep'), t('help_nutrition')
    ])
    
    with tab1:
        st.markdown("""
        **MENTAL HEALTH CRISIS - WHAT TO DO**
        
        **Call 988** - Suicide & Crisis Lifeline (24/7)
        
        **Warning Signs:**
        - Talking about wanting to die
        - Feeling hopeless
        - Withdrawing from others
        
        **What to do:**
        1. Ask directly if they're thinking about suicide
        2. Keep them safe
        3. Listen without judgment
        4. Help them call 988
        """)
    
    with tab2:
        st.markdown("""
        **MANAGING ANXIETY**
        
        **4-7-8 Breathing Technique:**
        - Inhale for 4 seconds
        - Hold for 7 seconds
        - Exhale for 8 seconds
        - Repeat 5 times
        
        **Visualization:** See yourself succeeding
        **Cue words:** "Calm", "Focus", "Breathe"
        """)
    
    with tab3:
        st.markdown("""
        **GYM IMPROVEMENT**
        
        **Progressive Overload:** Increase 5-10% weekly
        **Key Exercises:** Squats, deadlifts, bench press, rows
        **Rest:** 2-3 minutes between heavy sets
        
        **Signs of Overtraining:** Fatigue, poor sleep, irritability
        """)
    
    with tab4:
        st.markdown("""
        **PERFORMANCE OPTIMIZATION**
        
        **Daily Habits:**
        - 8+ hours sleep
        - Visualization (5 min)
        - Proper nutrition
        - Recovery work
        
        **Weekly Review:** 1 thing good, 1 to improve
        """)
    
    with tab5:
        st.markdown("""
        **SLEEP & RECOVERY**
        
        **Athlete Sleep Needs:** 8-10 hours
        **Optimization:**
        - No screens 1 hour before bed
        - Cool, dark room
        - Consistent schedule
        
        **Power Nap:** 20 min after lunch
        """)
    
    with tab6:
        st.markdown("""
        **SPORTS NUTRITION**
        
        **The Athlete's Plate:**
        - 50% carbs (rice, pasta, oats)
        - 25% protein (chicken, fish, eggs)
        - 25% vegetables
        
        **Post-Training (within 30 min):** Protein + carbs
        **Hydration:** 2-3 liters daily
        """)

# Footer
st.markdown("---")
st.caption("EmotionSport AI - Professional Athlete Monitoring System")
