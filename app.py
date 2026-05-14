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

# ============================================
# PAGE CONFIG
# ============================================
st.set_page_config(page_title="EmotionSport AI", page_icon="🧠", layout="wide")

# ============================================
# SESSION STATE INITIALIZATION
# ============================================
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
if 'show_help' not in st.session_state:
    st.session_state.show_help = False

# ============================================
# TRANSLATIONS
# ============================================
def t(key):
    texts = {
        'en': {
            # General
            'title': 'EmotionSport AI',
            'subtitle': 'Professional Athlete Emotion & Wellness Monitoring System',
            'login': 'Login',
            'logout': 'Logout',
            'username': 'Username / Email',
            'password': 'Password',
            'role': 'Role',
            'athlete': 'Athlete',
            'coach': 'Coach',
            'psychologist': 'Psychologist',
            'welcome': 'Welcome back',
            # Dashboard
            'dashboard': 'Wellness Dashboard',
            'emotion': 'Detected Emotion',
            'fatigue': 'Fatigue Level',
            'motivation': 'Motivation Level',
            'confidence': 'Confidence',
            'risk_level': 'Risk Level',
            'low_risk': 'Low Risk',
            'moderate_risk': 'Moderate Risk',
            'high_risk': 'High Risk',
            # Input
            'enter_text': 'Enter athlete text (interview, social media, self-report):',
            'predict': 'Analyze Emotion & Risk',
            # Interventions
            'interventions': 'Recommended Interventions',
            'crisis_intervention': '🚨 CRISIS INTERVENTION PROTOCOL',
            'anxiety_intervention': '😰 Anxiety Management Guide',
            'fatigue_intervention': '😴 Fatigue Recovery Protocol',
            'frustration_intervention': '😤 Frustration Management',
            'motivation_intervention': '🔥 Motivation Booster',
            'normal_intervention': '✅ Keep Going - Maintenance Mode',
            # Help sections
            'help_title': '📚 Athlete Support Center',
            'help_crisis': '🚨 Mental Health Crisis',
            'help_anxiety': '😰 Managing Anxiety',
            'help_gym': '💪 Gym & Training Improvement',
            'help_performance': '⚡ Performance Optimization',
            'help_sleep': '😴 Sleep & Recovery',
            'help_nutrition': '🥗 Sports Nutrition',
            # History
            'history': 'Recent History',
            'no_history': 'No predictions yet',
            'stats': 'Statistics',
            # Actions
            'view_guide': 'View Full Guide',
            'contact_coach': 'Contact Coach',
            'contact_psychologist': 'Contact Psychologist',
            'emergency': 'Emergency',
        },
        'es': {
            # General
            'title': 'EmotionSport AI',
            'subtitle': 'Sistema Profesional de Monitoreo de Emociones y Bienestar',
            'login': 'Iniciar Sesión',
            'logout': 'Cerrar Sesión',
            'username': 'Usuario / Email',
            'password': 'Contraseña',
            'role': 'Rol',
            'athlete': 'Atleta',
            'coach': 'Entrenador',
            'psychologist': 'Psicólogo',
            'welcome': 'Bienvenido',
            # Dashboard
            'dashboard': 'Panel de Bienestar',
            'emotion': 'Emoción Detectada',
            'fatigue': 'Nivel de Fatiga',
            'motivation': 'Nivel de Motivación',
            'confidence': 'Confianza',
            'risk_level': 'Nivel de Riesgo',
            'low_risk': 'Riesgo Bajo',
            'moderate_risk': 'Riesgo Moderado',
            'high_risk': 'Riesgo Alto',
            # Input
            'enter_text': 'Ingrese texto del atleta (entrevista, redes sociales, auto-reporte):',
            'predict': 'Analizar Emoción y Riesgo',
            # Interventions
            'interventions': 'Intervenciones Recomendadas',
            'crisis_intervention': '🚨 PROTOCOLO DE INTERVENCIÓN EN CRISIS',
            'anxiety_intervention': '😰 Guía para Manejar la Ansiedad',
            'fatigue_intervention': '😴 Protocolo de Recuperación de Fatiga',
            'frustration_intervention': '😤 Manejo de la Frustración',
            'motivation_intervention': '🔥 Impulso de Motivación',
            'normal_intervention': '✅ Mantenerse - Modo de Mantenimiento',
            # Help sections
            'help_title': '📚 Centro de Apoyo al Atleta',
            'help_crisis': '🚨 Crisis de Salud Mental',
            'help_anxiety': '😰 Manejo de la Ansiedad',
            'help_gym': '💪 Mejora en Gimnasio y Entrenamiento',
            'help_performance': '⚡ Optimización del Rendimiento',
            'help_sleep': '😴 Sueño y Recuperación',
            'help_nutrition': '🥗 Nutrición Deportiva',
            # History
            'history': 'Historial Reciente',
            'no_history': 'Sin predicciones aún',
            'stats': 'Estadísticas',
            # Actions
            'view_guide': 'Ver Guía Completa',
            'contact_coach': 'Contactar Entrenador',
            'contact_psychologist': 'Contactar Psicólogo',
            'emergency': 'Emergencia',
        }
    }
    return texts[st.session_state.lang][key]

# ============================================
# LANGUAGE SELECTOR
# ============================================
col_lang1, col_lang2, col_lang3 = st.columns([1, 1, 8])
with col_lang1:
    if st.button('🇬🇧 EN'):
        st.session_state.lang = 'en'
        st.rerun()
with col_lang2:
    if st.button('🇪🇸 ES'):
        st.session_state.lang = 'es'
        st.rerun()

st.title(t('title'))
st.markdown(f"### {t('subtitle')}")

# ============================================
# LOGIN SYSTEM
# ============================================
if not st.session_state.logged_in:
    with st.sidebar:
        st.markdown(f"### 🔐 {t('login')}")
        username = st.text_input(t('username'))
        password = st.text_input(t('password'), type="password")
        role = st.selectbox(t('role'), [t('athlete'), t('coach'), t('psychologist')])
        
        if st.button(t('login'), type="primary"):
            if username and password:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.session_state.user_role = role
                st.rerun()
            else:
                st.error("Please enter username and password")
    
    st.info("👈 Please login to access the emotion monitoring system")
    st.stop()

# ============================================
# SIDEBAR - USER INFO & QUICK ACTIONS
# ============================================
with st.sidebar:
    st.markdown(f"### 👋 {t('welcome')}, {st.session_state.username}!")
    st.markdown(f"**{t('role')}:** {st.session_state.user_role}")
    
    st.markdown("---")
    
    # Quick action buttons
    st.markdown("### ⚡ Quick Actions")
    if st.button("📞 " + t('contact_coach'), use_container_width=True):
        st.info("Coach notification sent (demo)")
    if st.button("🧠 " + t('contact_psychologist'), use_container_width=True):
        st.info("Psychologist notified (demo)")
    if st.button("🚨 " + t('emergency'), use_container_width=True, type="secondary"):
        st.error("🚨 EMERGENCY: Call 911 or 988 immediately")
    
    st.markdown("---")
    
    if st.button("🚪 " + t('logout'), use_container_width=True):
        st.session_state.logged_in = False
        st.session_state.username = None
        st.rerun()

# ============================================
# LOAD DATA & MODELS
# ============================================
@st.cache_data
def load_data():
    df = pd.read_csv('Emotion_Dataset_with_Self-Perceived_Performance12345.csv')
    df['main_emotion'] = df['main_emotion'].fillna('Unknown')
    df['text'] = df['text'].fillna('').astype(str)
    df = df[df['text'].str.strip() != '']
    
    # Create fatigue_level
    if 'fatigue_level' not in df.columns:
        fatigue_map = {
            'fatigue': 5, 'anxiety': 4, 'frustration': 4, 'insecurity': 3,
            'motivation': 3, 'confidence': 2, 'resilience': 2, 'euphoria': 1
        }
        df['fatigue_level'] = df['main_emotion'].str.lower().map(fatigue_map).fillna(3)
    
    # Create motivation_level
    if 'motivation_level' not in df.columns:
        motivation_map = {
            'motivation': 5, 'euphoria': 5, 'confidence': 4, 'resilience': 4,
            'frustration': 2, 'anxiety': 2, 'fatigue': 2, 'insecurity': 2
        }
        df['motivation_level'] = df['main_emotion'].str.lower().map(motivation_map).fillna(3)
    
    return df

@st.cache_resource
def train_models(df):
    emotion_encoder = LabelEncoder()
    y_emotion = emotion_encoder.fit_transform(df['main_emotion'])
    tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    X_tfidf = tfidf.fit_transform(df['text'])
    emotion_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    emotion_model.fit(X_tfidf, y_emotion)
    fatigue_model = RandomForestRegressor(n_estimators=100, random_state=42)
    fatigue_model.fit(X_tfidf, df['fatigue_level'])
    motivation_model = RandomForestRegressor(n_estimators=100, random_state=42)
    motivation_model.fit(X_tfidf, df['motivation_level'])
    return emotion_model, tfidf, emotion_encoder, fatigue_model, motivation_model

# Crisis keywords
CRISIS_WORDS = {
    'suicide': ['suicide', 'kill myself', 'end my life', 'want to die', 'self harm', 'self-harm'],
    'depression': ['depression', 'depressed', 'hopeless', 'worthless', 'numb', 'empty'],
    'severe_anxiety': ['panic', 'terrified', 'scared to death', 'cant breathe'],
    'self_harm': ['cut myself', 'hurt myself']
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

# Sidebar stats
with st.sidebar:
    st.markdown("---")
    st.markdown(f"### 📊 {t('stats')}")
    st.metric("Total Samples", len(df))
    st.metric("Emotion Classes", df['main_emotion'].nunique())

# ============================================
# MAIN CONTENT - TWO COLUMNS
# ============================================
col_main, col_dash = st.columns([2, 1])

with col_main:
    st.markdown(f"### 📝 {t('enter_text')}")
    user_input = st.text_area("", height=120, key="text_input")
    
    if st.button(f"🔮 {t('predict')}", type="primary", use_container_width=True):
        if user_input.strip():
            # Crisis check
            crisis, crisis_type = check_crisis(user_input)
            
            if crisis:
                st.error(f"""
                🚨🚨🚨 **{t('crisis_intervention')}** 🚨🚨🚨
                
                **Detected: {crisis_type.upper()}**
                
                **IMMEDIATE ACTIONS:**
                1. 🛑 Do NOT leave athlete alone
                2. 📞 Call 911 or 988 immediately
                3. 🧠 Contact team psychologist
                4. 🛡️ Remove from all activities
                
                **Resources:**
                - Emergency: 911
                - Crisis Lifeline: 988 (24/7)
                - Crisis Text Line: Text HOME to 741741
                """)
            
            # ML Predictions
            input_vec = tfidf.transform([user_input])
            emotion_pred = emotion_model.predict(input_vec)[0]
            emotion = emotion_encoder.inverse_transform([emotion_pred])[0]
            proba = emotion_model.predict_proba(input_vec)[0]
            confidence = max(proba) * 100
            fatigue_score = fatigue_model.predict(input_vec)[0]
            motivation_score = motivation_model.predict(input_vec)[0]
            
            # Risk level
            if crisis or fatigue_score >= 4.5:
                risk_level = "🔴 High Risk"
                risk_color = "red"
            elif fatigue_score >= 3.5 or emotion in ['Anxiety', 'Frustration']:
                risk_level = "🟡 Moderate Risk"
                risk_color = "orange"
            else:
                risk_level = "🟢 Low Risk"
                risk_color = "green"
            
            # Display metrics
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            with col_m1:
                st.metric(t('emotion'), emotion, f"{confidence:.0f}%")
            with col_m2:
                st.metric(t('fatigue'), f"{fatigue_score:.1f}/5", 
                         delta="High" if fatigue_score > 3.5 else "Normal")
            with col_m3:
                st.metric(t('motivation'), f"{motivation_score:.1f}/5",
                         delta="Good" if motivation_score > 3 else "Low")
            with col_m4:
                st.metric(t('risk_level'), risk_level)
            
            st.markdown("---")
            st.markdown(f"### 📋 {t('interventions')}")
            
            # Detailed interventions based on condition
            if crisis:
                st.markdown(f"**{t('crisis_intervention')}**")
                with st.expander("📞 Crisis Resources & Help", expanded=True):
                    st.markdown("""
                    **Emergency Contacts:**
                    - **988** - Suicide & Crisis Lifeline (24/7, confidential)
                    - **911** - Emergency Services
                    - **Text HOME to 741741** - Crisis Text Line
                    
                    **What to do NOW:**
                    1. Stay with the person - do not leave them alone
                    2. Remove any means of self-harm
                    3. Listen without judgment
                    4. Encourage professional help
                    5. Follow up within 24 hours
                    """)
            
            elif fatigue_score >= 4:
                st.markdown(f"**{t('fatigue_intervention')}**")
                col_int1, col_int2 = st.columns(2)
                with col_int1:
                    st.markdown("""
                    **⚠️ HIGH FATIGUE DETECTED**
                    
                    **Immediate Actions:**
                    - 🛑 Mandatory rest day
                    - 📉 Reduce training load by 50%
                    - 😴 Aim for 8-10 hours sleep
                    - 💧 Hydrate properly
                    
                    **Recovery Protocol:**
                    1. Active recovery (light walk)
                    2. Contrast bath (hot/cold)
                    3. Compression garments
                    4. Massage therapy if available
                    """)
                with col_int2:
                    st.markdown("""
                    **📊 Monitor These Signs:**
                    - Morning heart rate elevated >10%
                    - Poor sleep quality
                    - Irritability
                    - Lack of motivation
                    - Muscle soreness lasting >48h
                    
                    **Return to Training:**
                    Start at 50% volume, increase 10% daily if recovered
                    """)
            
            elif emotion == 'Anxiety' and confidence > 70:
                st.markdown(f"**{t('anxiety_intervention')}**")
                col_int1, col_int2 = st.columns(2)
                with col_int1:
                    st.markdown("""
                    **😰 Pre-Match/Competition Anxiety**
                    
                    **Immediate Techniques:**
                    - 🌬️ **Box Breathing:** 4 sec in → hold 4 → out 4 → hold 4
                    - 👁️ **Visualization:** See yourself succeeding
                    - 🎵 **Calming Playlist:** 60-80 BPM music
                    - 📝 **Journaling:** Write down worries, then set aside
                    
                    **30-Minute Pre-Performance Routine:**
                    1. Deep breathing (5 min)
                    2. Positive self-talk (2 min)
                    3. Visualization (5 min)
                    4. Light activation (10 min)
                    5. Focus on controllable factors
                    """)
                with col_int2:
                    st.markdown("""
                    **📅 Long-term Management:**
                    - 🧘 Daily mindfulness (10 min)
                    - 📖 Read "The Champion's Mind"
                    - 🎯 Set process goals (not outcome)
                    - 💬 Talk to sports psychologist
                    - 🏃‍♂️ Consistent pre-performance routine
                    
                    **Signs You Need Help:**
                    - Racing heart before easy practices
                    - Avoiding competition
                    - Physical symptoms (nausea, shaking)
                    - Sleep problems night before
                    """)
            
            elif emotion == 'Frustration' and confidence > 70:
                st.markdown(f"**{t('frustration_intervention')}**")
                col_int1, col_int2 = st.columns(2)
                with col_int1:
                    st.markdown("""
                    **😤 Managing Frustration on Field**
                    
                    **In-the-Moment Strategies:**
                    - ⏸️ **Timeout Signal:** Agree with coach to step aside
                    - 🌊 **Reset Routine:** Deep breath, adjust equipment
                    - 🎯 **Focus on Next Play:** Can't change past
                    - 💬 **Cue Words:** "Reset", "Next", "Breathe"
                    
                    **Between Plays:**
                    - Hands on knees, deep breath
                    - Positive self-talk
                    - Visualize successful execution
                    - Shake it off physically
                    """)
                with col_int2:
                    st.markdown("""
                    **📊 Post-Session Reflection:**
                    - What triggered frustration?
                    - How did I respond?
                    - What would I do differently?
                    - One thing I did well
                    
                    **Weekly Practice:**
                    - 🧘 10 min meditation daily
                    - 📝 Journal frustrations
                    - 🎯 Set realistic expectations
                    - 💪 Focus on controllables
                    """)
            
            elif motivation_score <= 2:
                st.markdown(f"**{t('motivation_intervention')}**")
                col_int1, col_int2 = st.columns(2)
                with col_int1:
                    st.markdown("""
                    **🔥 5-Minute Motivation Boosters:**
                    
                    1. **Watch Highlights:** 2 min of your best moments
                    2. **Power Pose:** 2 min (hands on hips, chest out)
                    3. **Gratitude List:** Write 3 things going well
                    4. **Music:** Pump-up playlist
                    5. **Talk to Someone:** Teammate or coach
                    
                    **Weekly Habit Building:**
                    - Set 3 small daily wins
                    - Create a motivation journal
                    - Find an accountability partner
                    - Reward yourself for effort, not just results
                    """)
                with col_int2:
                    st.markdown("""
                    **🎯 Goal Setting System:**
                    
                    **SMART Goals:**
                    - **S**pecific: What exactly?
                    - **M**easurable: How to track?
                    - **A**chievable: Realistic?
                    - **R**elevant: Why important?
                    - **T**ime-bound: When by?
                    
                    **Example:**
                    "I will arrive 15 min early to every practice this week to work on dribbling"
                    
                    **Track your wins!** Progress = Motivation
                    """)
            
            else:
                st.markdown(f"**{t('normal_intervention')}**")
                col_int1, col_int2 = st.columns(2)
                with col_int1:
                    st.markdown("""
                    **✅ What's Working Well:**
                    - Emotional state is balanced
                    - Fatigue levels are manageable
                    - Motivation is stable
                    
                    **Recommendations:**
                    - 📊 Continue daily check-ins
                    - 🎯 Maintain current training load
                    - 💪 Focus on skill development
                    - 🧘 Keep recovery routines
                    """)
                with col_int2:
                    st.markdown("""
                    **📈 Optimization Tips:**
                    - Add one new drill this week
                    - Try a new recovery method
                    - Set a small personal record
                    - Help a teammate with a skill
                    
                    **Keep the momentum going!**
                    """)
            
            # Save to history
            st.session_state.history.insert(0, {
                'text': user_input[:80],
                'emotion': emotion,
                'fatigue': round(float(fatigue_score), 1),
                'motivation': round(float(motivation_score), 1),
                'confidence': round(confidence, 1),
                'crisis': crisis,
                'risk': risk_level,
                'time': datetime.now().strftime('%H:%M')
            })
            st.session_state.history = st.session_state.history[:10]
            
        else:
            st.warning("Please enter some text to analyze")

# ============================================
# RIGHT COLUMN - DASHBOARD & HISTORY
# ============================================
with col_dash:
    st.markdown(f"### 📈 {t('dashboard')}")
    
    # Current status gauge
    if st.session_state.history:
        last = st.session_state.history[0]
        st.markdown(f"""
        **Current Status:**
        - 🎭 Emotion: **{last['emotion']}** ({last['confidence']}%)
        - 😴 Fatigue: {last['fatigue']}/5
        - 🔥 Motivation: {last['motivation']}/5
        - {last['risk']}
        """)
    
    # Emotion distribution chart
    emotion_counts = df['main_emotion'].value_counts().head(6)
    fig = px.bar(x=emotion_counts.values, y=emotion_counts.index, orientation='h',
                 title="Training Data Distribution")
    fig.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)
    
    # History
    st.markdown(f"### 📋 {t('history')}")
    if st.session_state.history:
        for h in st.session_state.history[:5]:
            crisis_icon = "🚨 " if h.get('crisis', False) else ""
            st.markdown(f"**{h['time']}** - {crisis_icon}{h['emotion']}")
            st.markdown(f"😴 {h['fatigue']}/5  🔥 {h['motivation']}/5")
            st.caption(f"{h['text']}...")
            st.markdown("---")
    else:
        st.info(t('no_history'))

# ============================================
# HELP SECTION - EXPANDABLE GUIDE
# ============================================
st.markdown("---")
with st.expander(f"📚 {t('help_title')} - Click to expand", expanded=False):
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        t('help_crisis'), t('help_anxiety'), t('help_gym'), 
        t('help_performance'), t('help_sleep'), t('help_nutrition')
    ])
    
    with tab1:
        st.markdown("""
        ### 🚨 MENTAL HEALTH CRISIS - WHAT TO DO
        
        **If you or someone you know is in immediate danger:**
        - **CALL 911** (Emergency Services)
        - **CALL 988** (Suicide & Crisis Lifeline) - 24/7, free, confidential
        - **Text HOME to 741741** - Crisis Text Line
        
        **Warning Signs:**
        - Talking about wanting to die or kill oneself
        - Looking for ways to kill oneself
        - Talking about feeling hopeless or having no reason to live
        - Talking about being a burden to others
        - Increasing alcohol or drug use
        - Withdrawing from activities
        - Sleeping too little or too much
        
        **What to do:**
        1. Ask directly: "Are you thinking about killing yourself?"
        2. Keep them safe - remove dangerous items
        3. Be there - listen without judgment
        4. Help them connect to professional help
        5. Follow up - check in within 48 hours
        
        **Resources:**
        - National Suicide Prevention Lifeline: **988**
        - Crisis Text Line: **Text HOME to 741741**
        - The Trevor Project (LGBTQ): **1-866-488-7386**
        - Veterans Crisis Line: **1-800-273-8255 (Press 1)**
        """)
    
    with tab2:
        st.markdown("""
        ### 😰 MANAGING PRE-COMPETITION ANXIETY
        
        **Understanding Anxiety:**
        Anxiety is normal! Even Olympians feel it. The goal isn't to eliminate anxiety, but to channel it productively.
        
        **The 4-7-8 Breathing Technique (2 minutes):**
        1. **Inhale** through nose for 4 seconds
        2. **Hold** breath for 7 seconds
        3. **Exhale** through mouth for 8 seconds
        4. Repeat 4-5 times
        
        **Visualization Script (5 minutes):**
        > "Close your eyes. See yourself walking onto the field/court. Feel your feet on the ground. Hear the crowd. See yourself executing perfectly - your technique is smooth, your mind is clear. You've done this a thousand times in practice. You belong here. Take a deep breath. Open your eyes. You're ready."
        
        **Pre-Competition Routine:**
        - **Night before:** Lay out all equipment, write down 3 things you control
        - **Morning of:** Eat normally, arrive early, listen to calming music
        - **1 hour before:** Light warm-up, breathing exercises
        - **30 min before:** Positive self-talk, visualization
        - **10 min before:** Final breathing check, cue words ready
        
        **Cue Words to Use:**
        - "Calm" - "Focus" - "Breathe" - "Process" - "Execute"
        
        **When to seek help:**
        - Anxiety interferes with sleep for multiple nights
        - Physical symptoms before every competition
        - Avoiding competitions or practice
        - Panic attacks
        """)
    
    with tab3:
        st.markdown("""
        ### 💪 GYM & TRAINING IMPROVEMENT GUIDE
        
        **Progressive Overload Principle:**
        Increase by 5-10% per week, not 20-30%. Patience = gains.
        
        **Weekly Training Split (Example):**
        | Day | Focus | Intensity |
        |-----|-------|-----------|
        | Mon | Upper Body + Cardio | 70% |
        | Tue | Lower Body + Agility | 75% |
        | Wed | Active Recovery | 50% |
        | Thu | Power + Explosiveness | 80% |
        | Fri | Full Body + Skills | 75% |
        | Sat | Match Simulation | 85% |
        | Sun | Complete Rest | 0% |
        
        **Key Exercises by Sport:**
        
        **Football/Soccer:**
        - Squats (strength)
        - Lunges (stability)
        - Box jumps (explosiveness)
        - Nordic curls (hamstring protection)
        
        **Basketball:**
        - Power cleans (vertical jump)
        - Bulgarian split squats (balance)
        - Medicine ball throws (core)
        
        **Tennis:**
        - Rotational exercises
        - Lateral lunges
        - Shoulder stability work
        
        **Recovery Between Sets:**
        - Strength: 2-3 min rest
        - Hypertrophy: 60-90 sec rest
        - Endurance: 30-45 sec rest
        
        **Signs of Overtraining:**
        - Persistent fatigue
        - Decreased performance
        - Irritability
        - Trouble sleeping
        - Frequent illness
        
        **If you see these signs → Take 2-3 days of light work**
        """)
    
    with tab4:
        st.markdown("""
        ### ⚡ PERFORMANCE OPTIMIZATION
        
        **The 3 Pillars of Elite Performance:**
        
        **1. MENTAL PREPARATION**
        - **Daily visualization:** 5 min imagining success
        - **Pre-performance routine:** Same actions before every practice/match
        - **Post-performance reflection:** 1 thing good, 1 to improve
        - **Mindfulness:** 10 min daily (apps: Calm, Headspace)
        
        **2. PHYSICAL PREPARATION**
        - **Sleep:** 8-10 hours for athletes
        - **Nutrition:** Eat within 30 min post-training
        - **Hydration:** 2-3 liters water daily, more with sweat
        - **Mobility:** 15 min daily dynamic stretching
        
        **3. TACTICAL PREPARATION**
        - **Video review:** 20 min weekly of own performance
        - **Opponent analysis:** Study next competitor
        - **Set piece practice:** Extra 10 min each practice
        - **Game scenarios:** Practice stressful situations
        
        **Weekly Performance Checklist:**
        - [ ] 8+ hours sleep each night
        - [ ] 3 strength sessions
        - [ ] 2 conditioning sessions
        - [ ] Daily visualization
        - [ ] Video review completed
        - [ ] Nutrition on point
        - [ ] Recovery work done
        
        **Peaking for Competition:**
        - **2 weeks out:** High intensity, sport-specific
        - **1 week out:** Maintain intensity, reduce volume by 20%
        - **3 days out:** Tactical focus, light work
        - **1 day out:** Rest, visualization, easy walk-through
        - **Game day:** Trust your preparation
        """)
    
    with tab5:
        st.markdown("""
        ### 😴 SLEEP & RECOVERY GUIDE FOR ATHLETES
        
        **Why Sleep Matters:**
        - Muscle repair happens during deep sleep
        - Growth hormone released during sleep
        - Mental processing and learning consolidation
        - Immune system strengthening
        
        **Sleep Requirements by Age:**
        | Age | Recommended Hours |
        |-----|-------------------|
        | 18-25 | 7-9 hours |
        | 26-64 | 7-9 hours |
        | Elite athletes | 8-10 hours |
        
        **Athlete Sleep Optimization Protocol:**
        
        **2 hours before bed:**
        - Stop intense training
        - Dim lights (use night mode on devices)
        - No caffeine (stop by 2pm)
        
        **1 hour before bed:**
        - Put away phone/tablet
        - Stretch or light yoga (10 min)
        - Journal (write down tomorrow's plan)
        
        **30 min before bed:**
        - Cool room (65-68°F / 18-20°C)
        - Blackout curtains or sleep mask
        - White noise or calming music
        
        **Napping Protocol:**
        - Power nap: 20 min (after lunch)
        - Recovery nap: 90 min (full cycle)
        - Best time: 1pm-3pm
        - Don't nap after 5pm
        
        **After Late Games:**
        - Cool down immediately
        - Eat within 30 min
        - Ice bath or contrast shower
        - Sleep in (adjust schedule next day)
        
        **Sleep Tracking:**
        Monitor these for 2 weeks to find your baseline:
        - Time to fall asleep
        - Night wakings
        - Morning energy (1-10 scale)
        - Practice energy (1-10 scale)
        """)
    
    with tab6:
        st.markdown("""
        ### 🥗 SPORTS NUTRITION FOR ATHLETES
        
        **The Athlete's Plate:**
        - 50% Complex carbohydrates (rice, pasta, potatoes, oats)
        - 25% Lean protein (chicken, fish, eggs, tofu, beans)
        - 25% Vegetables & fruits (variety of colors)
        
        **Pre-Training (2-3 hours before):**
        - Meal: Pasta with chicken, banana, water
        - Size: 400-600 calories
        - Avoid: Heavy fats, too much fiber
        
        **Pre-Training (30 min before):**
        - Snack: Banana, energy bar, toast with honey
        - Hydrate: 16-20 oz water
        
        **During Training (60+ min):**
        - Sports drink with electrolytes (every 15-20 min)
        - Energy gels if needed
        - Water between drinks
        
        **Post-Training (within 30 min - CRITICAL!):**
        - Protein shake + banana
        - Or chocolate milk (proven effective!)
        - Or Greek yogurt with berries
        
        **Post-Training meal (within 2 hours):**
        - Salmon with rice and vegetables
        - Or chicken wrap with avocado
        - Replenish with 24 oz water
        
        **Hydration Formula:**
        - Daily base: Body weight (lbs) ÷ 2 = oz water
        - During training: 7-10 oz every 15-20 min
        - After training: 24 oz for every pound lost
        
        **Check hydration:** Urine should be pale yellow
        
        **Supplements (consult doctor first):**
        - Vitamin D (if limited sun)
        - Omega-3 (anti-inflammatory)
        - Creatine (power sports)
        - Protein powder (convenience only)
        
        **Warning Signs (see a sports dietitian):**
        - Unexplained fatigue
        - Frequent illness
        - Poor recovery
        - Unintended weight loss
        - Disordered eating patterns
        """)

# Footer
st.markdown("---")
st.caption("EmotionSport AI | Emotion: ML Classifier | Fatigue: Random Forest | Motivation
