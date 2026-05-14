import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events
import platform
from gtts import gTTS
import base64
import io

# =========================================================
# CONFIGURACIÓN Y FUNCIONES DE APOYO
# =========================================================

st.set_page_config(page_title="Smart Home AI", page_icon="🏠", layout="wide")

def generar_voz(texto):
    """Genera y reproduce audio automáticamente en el navegador"""
    tts = gTTS(text=texto, lang='es')
    fp = io.BytesIO()
    tts.write_to_fp(fp)
    fp.seek(0)
    audio_b64 = base64.b64encode(fp.read()).decode()
    # HTML inyectado para reproducir audio sin mostrar controles
    audio_html = f"""
        <audio autoplay>
            <source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3">
        </audio>
    """
    st.markdown(audio_html, unsafe_allow_html=True)

# =========================================================
# ESTADOS DE LA SESIÓN
# =========================================================
if "door_open" not in st.session_state:
    st.session_state.door_open = False
if "kitchen_light" not in st.session_state:
    st.session_state.kitchen_light = False
if "recognized_person" not in st.session_state:
    st.session_state.recognized_person = ""

# =========================================================
# CSS / ESTILO FUTURISTA
# =========================================================
st.markdown("""
<style>
.stApp{ background-color:#0F172A; color:white; }
.main-title{ text-align:center; font-size:50px; font-weight:bold; color:#E9D5FF; }
.subtitle{ text-align:center; color:#CBD5E1; margin-bottom:30px; }
.room{ padding:30px; border-radius:20px; text-align:center; margin-bottom:20px; box-shadow:0px 4px 20px rgba(0,0,0,0.3); }
.room-off{ background-color:#1E293B; border:2px solid #334155; }
.room-on{ background-color:#FDE68A; color:black; border:2px solid #FBBF24; }
.door-open{ background-color:#86EFAC; color:black; border:2px solid #22C55E; }
.door-closed{ background-color:#7F1D1D; border:2px solid #EF4444; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">🏠 SMART HOME AI</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Reconocimiento Facial + Control por Voz</p>', unsafe_allow_html=True)

# =========================================================
# CARGAR MODELO
# =========================================================
@st.cache_resource
def load_my_model():
    return load_model("keras_model.h5")

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Error cargando modelo: {e}")

# =========================================================
# INTERFAZ VISUAL (CASA)
# =========================================================
col1, col2 = st.columns(2)

with col1:
    if st.session_state.door_open:
        st.markdown(f'<div class="room door-open"><h1>🚪</h1><h2>ABIERTA</h2><p>{st.session_state.recognized_person}</p></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="room door-closed"><h1>🚪</h1><h2>CERRADA</h2></div>', unsafe_allow_html=True)

with col2:
    if st.session_state.kitchen_light:
        st.markdown('<div class="room room-on"><h1>💡🍳</h1><h2>COCINA ON</h2></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="room room-off"><h1>💡</h1><h2>COCINA OFF</h2></div>', unsafe_allow_html=True)

# =========================================================
# LÓGICA DE RECONOCIMIENTO FACIAL
# =========================================================
st.divider()
img_file_buffer = st.camera_input("Escaneando rostro...")

if img_file_buffer is not None:
    img = Image.open(img_file_buffer).convert("RGB")
    img = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
    img_array = np.asarray(img)
    normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    prediction = model.predict(data)
    index = np.argmax(prediction)
    confidence = prediction[0][index]

    if confidence > 0.80:
        nombres = ["María José", "Camilo"] # Ajusta según tus etiquetas
        persona = nombres[index] if index < len(nombres) else "Desconocido"
        
        if st.session_state.recognized_person != persona:
            st.session_state.door_open = True
            st.session_state.recognized_person = persona
            st.success(f"Bienvenido/a {persona}")
            generar_voz(f"Bienvenida a casa {persona}. La puerta ha sido abierta.")
            st.balloons()
    else:
        st.error("Rostro no reconocido")

# =========================================================
# LÓGICA DE VOZ
# =========================================================
st.divider()
st.header("🎤 Control por Voz")
stt_button = Button(label="🎤 Presiona para hablar", width=250)
stt_button.js_on_event("button_click", CustomJS(code="""
    var recognition = new webkitSpeechRecognition();
    recognition.lang = 'es-ES';
    recognition.onresult = function(e) {
        var value = e.results[0][0].transcript;
        document.dispatchEvent(new CustomEvent("GET_TEXT", {detail: value}));
    }
    recognition.start();
"""))

result = streamlit_bokeh_events(stt_button, events="GET_TEXT", key="listen", refresh_on_update=False, override_height=75)

if result and "GET_TEXT" in result:
    comando = result.get("GET_TEXT").lower()
    st.info(f"Escuché: {comando}")

    if "encender cocina" in comando:
        st.session_state.kitchen_light = True
        generar_voz("Entendido, encendiendo la cocina")
    elif "apagar cocina" in comando:
        st.session_state.kitchen_light = False
        generar_voz("Cocinando luces apagadas")
    elif "abrir puerta" in comando:
        st.session_state.door_open = True
        generar_voz("Abriendo la puerta principal")
    elif "cerrar puerta" in comando:
        st.session_state.door_open = False
        generar_voz("Puerta cerrada, seguridad activada")
    
    st.rerun()

st.caption(f"Sistema Operativo: {platform.system()} | Python {platform.python_version()}")
