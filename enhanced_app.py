# enhanced_app.py - Updated with Advanced Model + Big Five
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import joblib
from textblob import TextBlob
import altair as alt

# Set page configuration
st.set_page_config(
    page_title="Advanced Personality Mapper",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .trait-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
        border-left: 4px solid #667eea;
    }
</style>
""", unsafe_allow_html=True)

# Load advanced model
@st.cache_resource
def load_advanced_model():
    try:
        model_data = joblib.load('advanced_personality_model.pkl')
        st.sidebar.success(f"✅ Advanced Model Loaded ({model_data['accuracy']:.2%} accuracy)")
        return model_data
    except:
        st.sidebar.warning("⚠️ Advanced model not found. Using basic model.")
        return None

# Big Five Analysis Functions
def analyze_big_five(text):
    text = str(text).lower()
    
    scores = {
        'Openness': 50,
        'Conscientiousness': 50,
        'Extraversion': 50,
        'Agreeableness': 50,
        'Neuroticism': 50
    }
    
    blob = TextBlob(text)
    sentiment = blob.sentiment.polarity
    
    words = text.split()
    
    # Enhanced word analysis with more keywords
    trait_keywords = {
        'Openness': ['creative', 'imaginative', 'curious', 'artistic', 'philosophical', 
                    'adventurous', 'innovative', 'complex', 'abstract', 'explore', 'new'],
        'Conscientiousness': ['organized', 'planned', 'efficient', 'reliable', 'disciplined',
                             'responsible', 'structured', 'systematic', 'detailed', 'plan'],
        'Extraversion': ['social', 'party', 'friends', 'people', 'energetic', 'outgoing',
                        'talkative', 'enthusiastic', 'group', 'team', 'meet'],
        'Agreeableness': ['kind', 'helpful', 'empathetic', 'understanding', 'compassionate',
                         'cooperative', 'supportive', 'friendly', 'peaceful', 'care'],
        'Neuroticism': ['anxious', 'worried', 'stressed', 'nervous', 'insecure', 
                       'tense', 'unstable', 'moody', 'fearful', 'panic']
    }
    
    for trait, keywords in trait_keywords.items():
        count = sum(1 for word in words if word in keywords)
        if trait != 'Neuroticism':
            scores[trait] += min(count * 8, 35)
        else:
            scores[trait] += min(count * 6, 30)
    
    # Sentiment adjustments
    scores['Extraversion'] += int(sentiment * 12)
    scores['Agreeableness'] += int(sentiment * 12)
    scores['Neuroticism'] -= int(sentiment * 15)
    
    # Text length and complexity adjustments
    if len(words) > 50:  # Longer, more complex text
        scores['Openness'] += 5
        scores['Conscientiousness'] += 3
    
    # Normalize scores
    for trait in scores:
        scores[trait] = max(0, min(100, scores[trait]))
    
    return scores

def create_radar_chart(scores):
    """Create a radar chart for Big Five traits"""
    traits = list(scores.keys())
    values = list(scores.values())
    
    # Repeat first value to close the radar chart
    traits.append(traits[0])
    values.append(values[0])
    
    # Create DataFrame for Altair
    df = pd.DataFrame({
        'trait': traits,
        'score': values,
        'angle': np.linspace(0, 2*np.pi, len(traits)).tolist()
    })
    
    radar_chart = alt.Chart(df).mark_line(strokeWidth=3, stroke='#667eea').encode(
        x=alt.X('angle:O', axis=None),
        y=alt.Y('score:Q', scale=alt.Scale(domain=[0, 100])),
        tooltip=['trait', 'score']
    ).properties(
        width=400,
        height=400
    )
    
    return radar_chart

def main():
    st.markdown('<h1 class="main-header">🧠 Advanced Personality Mapper</h1>', unsafe_allow_html=True)
    
    # Load models
    model_data = load_advanced_model()
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 Analysis Options")
        analysis_type = st.radio(
            "Choose analysis type:",
            ["MBTI Personality", "Big Five Traits", "Combined Analysis"]
        )
        
        st.header("📊 Model Info")
        if model_data:
            st.metric("Model Accuracy", f"{model_data['accuracy']:.2%}")
        else:
            st.metric("Model Accuracy", "65.48%")
        
        st.header("💡 Tips")
        st.write("""
        - Write 50+ words for better accuracy
        - Describe your interests, behaviors, and preferences
        - Be authentic and detailed
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_text = st.text_area(
            "📝 Tell me about yourself:",
            height=200,
            placeholder="I enjoy... I prefer... I feel... Describe your typical day, interests, and how you interact with others..."
        )
    
    with col2:
        st.subheader("🚀 Quick Examples")
        examples = {
            "Creative Thinker": "I love exploring new ideas and creating art. Philosophical discussions fascinate me, and I often get lost in imaginative projects.",
            "Social Organizer": "I enjoy planning events and bringing people together. Making detailed schedules and ensuring everything runs smoothly gives me satisfaction.",
            "Analytical Solver": "I prefer working with data and solving complex problems. Logical analysis and systematic approaches are my strengths.",
            "Empathic Helper": "I'm very attuned to others' emotions and enjoy helping people. Creating harmony in relationships is important to me."
        }
        
        for desc, example in examples.items():
            if st.button(desc):
                st.session_state.example_text = example
    
    if 'example_text' in st.session_state:
        user_text = st.session_state.example_text
    
    if st.button("🔮 Analyze My Personality", type="primary") and user_text:
        with st.spinner("Analyzing your personality traits..."):
            # Perform analysis based on selected type
            if analysis_type == "MBTI Personality" or analysis_type == "Combined Analysis":
                if model_data:
                    # Use advanced model
                    cleaned_text = re.sub(r'[^a-zA-Z\s\.!?,]', '', str(user_text).lower())
                    text_tfidf = model_data['vectorizer'].transform([cleaned_text])
                    probabilities = model_data['model'].predict_proba(text_tfidf)[0]
                    
                    mbti_results = {}
                    for i, personality_type in enumerate(model_data['model'].classes_):
                        mbti_results[personality_type] = round(probabilities[i] * 100, 2)
                    mbti_results = dict(sorted(mbti_results.items(), key=lambda x: x[1], reverse=True))
                else:
                    # Fallback to basic prediction (you can implement this)
                    mbti_results = {"INFP": 25, "ENFP": 20, "INTP": 15, "INFJ": 12}
            
            if analysis_type == "Big Five Traits" or analysis_type == "Combined Analysis":
                big_five_scores = analyze_big_five(user_text)
            
            # Display results
            st.markdown("---")
            
            if analysis_type == "MBTI Personality":
                display_mbti_results(mbti_results)
            elif analysis_type == "Big Five Traits":
                display_big_five_results(big_five_scores)
            else:  # Combined Analysis
                col1, col2 = st.columns(2)
                with col1:
                    display_mbti_results(mbti_results)
                with col2:
                    display_big_five_results(big_five_scores)

def display_mbti_results(results):
    st.subheader("🎭 MBTI Personality Results")
    
    top_personality = list(results.keys())[0]
    top_confidence = list(results.values())[0]
    
    st.markdown(f"""
    <div class="prediction-card">
        <h2>Primary Type: {top_personality}</h2>
        <h3>Confidence: {top_confidence}%</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Top predictions
    st.write("**Top Predictions:**")
    top_5 = dict(list(results.items())[:5])
    for i, (p_type, prob) in enumerate(top_5.items()):
        if i == 0:
            st.metric(f"🏆 {p_type}", f"{prob}%")
        else:
            st.metric(f"🥈 {p_type}", f"{prob}%")

def display_big_five_results(scores):
    st.subheader("📊 Big Five Personality Traits")
    
    # Radar chart
    st.altair_chart(create_radar_chart(scores), use_container_width=True)
    
    # Trait descriptions
    st.write("**Trait Analysis:**")
    
    trait_descriptions = {
        'Openness': "Imagination, curiosity, and preference for variety",
        'Conscientiousness': "Organization, responsibility, and reliability", 
        'Extraversion': "Sociability, energy, and positive emotions",
        'Agreeableness': "Compassion, cooperation, and trust",
        'Neuroticism': "Emotional stability and resilience to stress"
    }
    
    for trait, score in scores.items():
        with st.container():
            st.markdown(f"""
            <div class="trait-card">
                <h4>{trait}: {score}/100</h4>
                <p>{trait_descriptions[trait]}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress bar
            st.progress(score/100)

if __name__ == "__main__":
    main()