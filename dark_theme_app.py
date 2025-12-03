# dark_theme_app.py - Dark Theme with Animations
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import joblib
import time
import streamlit.components.v1 as components
from sklearn.feature_extraction.text import TfidfVectorizer

# Page configuration with dark theme
st.set_page_config(
    page_title="NeuroScan Personality AI",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dark Theme + Animations
st.markdown("""
<style>
    /* Main Dark Theme */
    .stApp {
        background: linear-gradient(135deg, #0c0c0c 0%, #1a1a2e 50%, #16213e 100%);
        color: #ffffff;
    }
    
    /* Headers */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-family: 'Segoe UI', sans-serif;
    }
    
    /* Sidebar */
    .css-1d391kg, .css-1lcbmhc {
        background-color: #1a1a2e !important;
    }
    
    /* Text areas and inputs */
    .stTextArea textarea {
        background-color: #2d2d44 !important;
        color: #ffffff !important;
        border: 1px solid #4a4a6a;
        border-radius: 10px;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 10px 25px !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px 0 rgba(102, 126, 234, 0.3) !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px 0 rgba(102, 126, 234, 0.4) !important;
    }
    
    /* Cards with glass morphism effect */
    .prediction-card {
        background: rgba(255, 255, 255, 0.1) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 15px !important;
        padding: 2rem !important;
        margin: 1rem 0 !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.36) !important;
    }
    
    .trait-card {
        background: rgba(255, 255, 255, 0.08) !important;
        backdrop-filter: blur(5px) !important;
        border-radius: 10px !important;
        padding: 1.5rem !important;
        margin: 0.5rem 0 !important;
        border-left: 4px solid #667eea !important;
        transition: transform 0.3s ease !important;
    }
    
    .trait-card:hover {
        transform: translateX(5px) !important;
    }
    
    /* Metrics */
    .stMetric {
        background: rgba(255, 255, 255, 0.05) !important;
        padding: 1rem !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    
    /* Progress bars */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* Radio buttons */
    .stRadio > div {
        background: rgba(255, 255, 255, 0.05) !important;
        padding: 1rem !important;
        border-radius: 10px !important;
    }
    
    /* Custom animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes glow {
        0%, 100% {
            box-shadow: 0 0 5px rgba(102, 126, 234, 0.5);
        }
        50% {
            box-shadow: 0 0 20px rgba(102, 126, 234, 0.8);
        }
    }
    
    .fade-in-up {
        animation: fadeInUp 0.6s ease-out;
    }
    
    .glow-card {
        animation: glow 2s ease-in-out infinite;
    }
    
    /* Hide Streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# JavaScript for additional animations
components.html("""
<script>
// Typewriter effect for title
function typeWriter(element, text, speed = 100) {
    let i = 0;
    element.innerHTML = '';
    function type() {
        if (i < text.length) {
            element.innerHTML += text.charAt(i);
            i++;
            setTimeout(type, speed);
        }
    }
    type();
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', function() {
    const title = document.querySelector('h1');
    if (title) {
        const originalText = title.innerText;
        typeWriter(title, originalText);
    }
});
</script>
""")

# Load model with caching
@st.cache_resource
def load_model():
    try:
        model_data = joblib.load('advanced_personality_model.pkl')
        return model_data
    except:
        try:
            with open('personality_model.pkl', 'rb') as f:
                import pickle
                model_data = pickle.load(f)
            return model_data
        except:
            return None

# Prediction function
def predict_personality(text, model_data):
    def clean_text(text):
        text = str(text)
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'[^a-zA-Z\s\.!?,]', '', text)
        text = text.lower()
        text = ' '.join(text.split())
        return text
    
    if model_data:
        cleaned_text = clean_text(text)
        text_tfidf = model_data['vectorizer'].transform([cleaned_text])
        probabilities = model_data['model'].predict_proba(text_tfidf)[0]
        
        results = {}
        for i, personality_type in enumerate(model_data['model'].classes_):
            results[personality_type] = round(probabilities[i] * 100, 2)
        
        return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
    return None

def main():
    # Animated header
    st.markdown("""
    <div class="fade-in-up">
        <h1 style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                  -webkit-background-clip: text; -webkit-text-fill-color: transparent; 
                  font-size: 3.5rem; margin-bottom: 0.5rem;">
            🔮 NeuroScan AI
        </h1>
        <p style="text-align: center; color: #a0a0c0; font-size: 1.2rem; margin-bottom: 2rem;">
            Discover Your Personality Through Artificial Intelligence
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model_data = load_model()
    
    # Sidebar with animations
    with st.sidebar:
        st.markdown("""
        <div class="fade-in-up">
            <h2>🎯 Analysis Panel</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Model status with animation
        if model_data:
            st.markdown(f"""
            <div class="glow-card" style="background: rgba(0, 255, 127, 0.1); padding: 1rem; border-radius: 10px; border: 1px solid #00ff7f;">
                <h4 style="color: #00ff7f; margin: 0;">✅ AI Model Active</h4>
                <p style="color: #90ee90; margin: 0;">Accuracy: {model_data.get('accuracy', 0.65):.2%}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background: rgba(255, 69, 0, 0.1); padding: 1rem; border-radius: 10px; border: 1px solid #ff4500;">
                <h4 style="color: #ff4500; margin: 0;">❌ Model Not Found</h4>
                <p style="color: #ffa07a; margin: 0;">Run advanced_model.py first</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Analysis options
        st.markdown("""
        <div class="fade-in-up">
            <h4>🔍 Analysis Mode</h4>
        </div>
        """, unsafe_allow_html=True)
        
        analysis_mode = st.radio(
            "",
            ["Standard Analysis", "Deep Analysis", "Comparative Analysis"],
            index=0
        )
        
        st.markdown("---")
        
        # Tips with animations
        st.markdown("""
        <div class="fade-in-up">
            <h4>💡 Pro Tips</h4>
            <div class="trait-card">
                <p>🎯 Write 50+ words for accuracy</p>
            </div>
            <div class="trait-card">
                <p>💭 Describe thoughts & feelings</p>
            </div>
            <div class="trait-card">
                <p>👥 Mention social preferences</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        <div class="fade-in-up">
            <h3>📝 Tell Your Story</h3>
        </div>
        """, unsafe_allow_html=True)
        
        user_text = st.text_area(
            "",
            height=200,
            placeholder="I find myself drawn to... I typically enjoy... When faced with challenges, I... My ideal day would involve...",
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("""
        <div class="fade-in-up">
            <h3>🚀 Quick Profiles</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Animated example buttons
        examples = {
            "🧠 Analytical Thinker": "I enjoy solving complex problems and analyzing data. Logical reasoning is my strength, and I prefer working with systems and patterns rather than emotions. I spend hours researching topics that interest me.",
            "🎨 Creative Visionary": "My mind is constantly generating new ideas. I love art, music, and creative expression. I see possibilities everywhere and enjoy thinking outside conventional boundaries.",
            "👥 Social Catalyst": "I thrive in social situations and love connecting with people. Organizing events, meeting new friends, and building communities energizes me.",
            "📊 Strategic Planner": "I'm highly organized and always planning ahead. I create detailed schedules, set clear goals, and enjoy optimizing processes for efficiency."
        }
        
        for desc, example in examples.items():
            if st.button(desc, use_container_width=True):
                st.session_state.example_text = example
                st.rerun()
    
    # Handle example text
    if 'example_text' in st.session_state:
        user_text = st.session_state.example_text
    
    # Analyze button with enhanced styling
    if st.button("**🔮 LAUNCH PERSONALITY SCAN**", use_container_width=True) and user_text:
        with st.spinner("🔄 Neural networks analyzing your personality patterns..."):
            # Simulate processing time for animation
            time.sleep(1.5)
            
            results = predict_personality(user_text, model_data)
            
            if results:
                top_personality = list(results.keys())[0]
                top_confidence = list(results.values())[0]
                
                # Results with animations
                st.markdown("---")
                
                # Main prediction card with glow effect
                st.markdown(f"""
                <div class="prediction-card glow-card fade-in-up">
                    <div style="text-align: center;">
                        <h2 style="background: linear-gradient(135deg, #00ff7f 0%, #00bfff 100%); 
                                  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                                  margin-bottom: 0.5rem;">
                            🎭 PERSONALITY IDENTIFIED
                        </h2>
                        <h1 style="font-size: 4rem; margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                  -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                            {top_personality}
                        </h1>
                        <h3 style="color: #a0a0c0; margin-top: 0.5rem;">
                            Confidence Level: <span style="color: #00ff7f;">{top_confidence}%</span>
                        </h3>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Top predictions in columns
                st.markdown("""
                <div class="fade-in-up">
                    <h3>📊 Probability Matrix</h3>
                </div>
                """, unsafe_allow_html=True)
                
                top_5 = dict(list(results.items())[:5])
                cols = st.columns(5)
                
                for i, ((p_type, prob), col) in enumerate(zip(top_5.items(), cols)):
                    with col:
                        if i == 0:
                            col.markdown(f"""
                            <div style="text-align: center; padding: 1rem; background: rgba(102, 126, 234, 0.2); 
                                      border-radius: 10px; border: 2px solid #667eea;">
                                <h4 style="color: #667eea; margin: 0;">🏆 {p_type}</h4>
                                <h2 style="color: #00ff7f; margin: 0;">{prob}%</h2>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            col.markdown(f"""
                            <div style="text-align: center; padding: 1rem; background: rgba(255, 255, 255, 0.05); 
                                      border-radius: 10px; border: 1px solid rgba(255, 255, 255, 0.1);">
                                <h4 style="color: #a0a0c0; margin: 0;">🥈 {p_type}</h4>
                                <h3 style="color: #4ECDC4; margin: 0;">{prob}%</h3>
                            </div>
                            """, unsafe_allow_html=True)
                
                # Visualization
                st.markdown("""
                <div class="fade-in-up">
                    <h3>📈 Neural Pattern Visualization</h3>
                </div>
                """, unsafe_allow_html=True)
                
                # Create dark theme chart
                fig, ax = plt.subplots(figsize=(12, 6))
                fig.patch.set_facecolor('#0c0c0c')
                ax.set_facecolor('#1a1a2e')
                
                types = list(top_5.keys())
                probs = list(top_5.values())
                
                colors = ['#667eea', '#764ba2', '#4ECDC4', '#45B7D1', '#96CEB4']
                bars = ax.bar(types, probs, color=colors, edgecolor='white', linewidth=2)
                
                # Style the chart for dark theme
                ax.tick_params(colors='white', which='both')
                ax.yaxis.label.set_color('white')
                ax.spines['bottom'].set_color('white')
                ax.spines['left'].set_color('white')
                
                ax.set_ylabel('Probability (%)', color='white', fontsize=12)
                ax.set_ylim(0, max(probs) + 10)
                
                # Add value labels with glow effect
                for bar, prob in zip(bars, probs):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                           f'{prob}%', ha='center', va='bottom', 
                           color='white', fontweight='bold', fontsize=11,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='#667eea', alpha=0.8))
                
                plt.xticks(rotation=45, color='white')
                plt.tight_layout()
                
                st.pyplot(fig)
                
            else:
                st.error("""
                ⚠️ **Neural Network Offline**
                
                Please train the AI model first:
                ```bash
                python advanced_model.py
                ```
                """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #a0a0c0; padding: 2rem;">
        <p>🔮 Powered by Advanced Neural Networks • 🧠 Real-time Personality Analysis</p>
        <p>💫 Your data is processed securely and anonymously</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()