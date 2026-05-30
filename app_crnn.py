import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
import io
from streamlit_mic_recorder import mic_recorder
import auth_manager as auth

st.set_page_config(page_title="NeuroVoice AI", page_icon="🧠", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("parkinson_crnn_model.h5")

model = load_model()

if 'user' not in st.session_state: st.session_state.user = None
if 'screen' not in st.session_state: st.session_state.screen = 'login'


def preprocess_audio_bytes(audio_bytes):
    try:
        import imageio_ffmpeg
        import subprocess
        import tempfile
        import os
        import librosa
        import numpy as np
        
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp_in:
            temp_in.write(audio_bytes)
            temp_in_path = temp_in.name
            
        temp_out_path = temp_in_path + ".wav"
        
        subprocess.run([
            ffmpeg_exe, 
            "-y",             
            "-i", temp_in_path, 
            temp_out_path     
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        y, sr = librosa.load(temp_out_path, sr=22050, duration=3)
        
   
        os.remove(temp_in_path)
        os.remove(temp_out_path)
        
        
        target_length = sr * 3  
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))
        elif len(y) > target_length:
            y = y[:target_length]
            
        
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        delta_mfcc = librosa.feature.delta(mfcc)
        delta2_mfcc = librosa.feature.delta(mfcc, order=2)
        
        features = np.stack([mfcc, delta_mfcc, delta2_mfcc], axis=-1)
        features = np.transpose(features, (1, 0, 2))
        
        return features[np.newaxis, ...]
        
    except Exception as e:
        import streamlit as st
        st.error(f"Error crítico en el preprocesamiento de audio: {e}")
        return None



if st.session_state.screen == 'login':
    st.title("🧬 NeuroVoice AI")
    tab1, tab2 = st.tabs(["Log in", "Sign up"])
    
    with tab1:
        email = st.text_input("Email", key="l_email")
        pw = st.text_input("Password", type="password", key="l_pw")
        if st.button("Log in", use_container_width=True):
            user, msg = auth.login_user(email, pw)
            if user:
                st.session_state.user = user
                st.session_state.screen = 'home'
                st.experimental_rerun()()
            else: st.error(msg)
            
    with tab2:
        name = st.text_input("Full name")
        new_email = st.text_input("Email")
        new_pw = st.text_input("Password", type="password")
        if st.button("Create an account", use_container_width=True):
            success, msg = auth.register_user(name, new_email, new_pw)
            if success: st.success(msg)
            else: st.error(msg)

elif st.session_state.screen == 'home':
    user = st.session_state.user
    st.title(f"👋 ¡Hello, {user['name']}!")
    
    # Gráfica de evolución
    if user['history']:
        st.subheader("Your Stability Progress")
        data = [h['average'] for h in user['history']]
        st.line_chart(data)
    else:
        st.info("You haven't registered yet. Go ahead and complete your first check-up!")

    if st.button("Start a New Dual Check", use_container_width=True, type="primary"):
        st.session_state.screen = 'test'
        st.experimental_rerun()
        
    if st.button("My Profile and Settings", use_container_width=True):
        st.session_state.screen = 'profile'
        st.experimental_rerun()


elif st.session_state.screen == 'test':
    st.title("Robust Dual Capture")
    st.write("Please make both recordings before proceeding with the analysis.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Test 1: Vocal A")
        st.info("Hold the vowel “A” for 3 seconds.")
        audio1 = mic_recorder(start_prompt="🔴 Record 'A'", stop_prompt="⏹️ Stop Test 1", key='v1')
        if audio1 and 'bytes' in audio1:
            st.success("✅ Audio 1 received successfully")
            
    with col2:
        st.subheader("Test 2: Reading")
        st.info("Read the sentence aloud: 'The sun rises from the east.'.")
        audio2 = mic_recorder(start_prompt="🔴 Record sentence", stop_prompt="⏹️ Stop Test 2", key='v2')
        if audio2 and 'bytes' in audio2:
            st.success("✅ Audio 2 received successfully")

    st.markdown("---")

    if audio1 and audio2 and ('bytes' in audio1) and ('bytes' in audio2):
        if st.button("Analyse Both Audio Files and Generate a Diagnosis", use_container_width=True, type="primary"):
            with st.spinner("The CRNN neural network is averaging the spectral features..."):
                
                in1 = preprocess_audio_bytes(audio1['bytes'])
                in2 = preprocess_audio_bytes(audio2['bytes'])
                
                
                if in1 is not None and in2 is not None:
                    p1 = model.predict(in1)[0][0]
                    p2 = model.predict(in2)[0][0]
                    
                    new_data = auth.save_test_result(st.session_state.user['email'], p1, p2)
                    st.session_state.user = new_data
                    st.session_state.last_results = (p1, p2)
                    st.session_state.screen = 'results'
                    st.experimental_rerun()
                else:
                    st.error("❌ One of the audio files could not be decoded. Please try recording again in a quieter environment.")
    else:
        st.warning("⚠️ Waiting for you to complete both recordings so that we can activate dual diagnostics.")
    
    if st.button("⬅️ Cancel and Return to Home", use_container_width=True):
        st.session_state.screen = 'home'
        st.experimental_rerun()

elif st.session_state.screen == 'results':
    p1, p2 = st.session_state.last_results
    # avg = (p1 + p2) / 2
    st.title("📊 Diagnostic Results")
    
    st.write("Based on an acoustic analysis of both recordings:")

    # if avg > 0.5:
    if p1 > 0.5 or p2 > 0.5:
        st.warning("Voice patterns consistent with Parkinson's disease have been detected. A medical consultation is recommended.")
    else:
        st.success("Your voice shows consistent and healthy patterns.")
        
    # st.write(f"Vocal: {round((1-p1)*100)}% | Lectura: {round((1-p2)*100)}%")
    
    if st.button("Back to Home", use_container_width=True):
        st.session_state.screen = 'home'
        st.experimental_rerun()

elif st.session_state.screen == 'profile':
    st.title("👤 My Profile")
    user = st.session_state.user
    st.write(f"**Name:** {user['name']}")
    st.write(f"**Email:** {user['email']}")
    
    if st.button("Log out", type="secondary"):
        st.session_state.user = None
        st.session_state.screen = 'login'
        st.experimental_rerun()
        
    if st.button("Back"):
        st.session_state.screen = 'home'
        sst.experimental_rerun()
