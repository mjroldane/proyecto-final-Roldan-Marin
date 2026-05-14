import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import os
import io
import base64
import platform
 
# =========================================================
# CONFIGURACIÓN DE PÁGINA
# =========================================================
 
st.set_page_config(
    page_title="Smart Home AI",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="collapsed"
)
 
# =========================================================
# CARGA DEL MODELO — compatible con TensorFlow moderno
# =========================================================
 
@st.cache_resource
def load_my_model():
    """
    Intenta cargar el modelo con múltiples estrategias de compatibilidad.
    Esto resuelve el error DepthwiseConv2D 'groups' de Teachable Machine.
    """
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, "keras_model.h5")
 
    # Estrategia 1: tf-keras (paquete legacy recomendado para modelos Teachable Machine)
    try:
        import tf_keras
        model = tf_keras.models.load_model(model_path, compile=False)
        return model, "tf_keras"
    except Exception:
        pass
 
    # Estrategia 2: tensorflow.keras con compile=False
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model(model_path, compile=False)
        return model, "tf.keras"
    except Exception:
        pass
 
    # Estrategia 3: keras standalone
    try:
        from keras.models import load_model
        model = load_model(model_path, compile=False)
        return model, "keras"
    except Exception as e:
        raise RuntimeError(f"No se pudo cargar el modelo con ninguna estrategia: {e}")
 
# =========================================================
# FUNCIÓN DE VOZ
# =========================================================
 
def generar_voz(texto):
    try:
        from gtts import gTTS
        tts = gTTS(text=texto, lang='es')
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        audio_b64 = base64.b64encode(fp.read()).decode()
        st.markdown(
            f'<audio autoplay><source src="data:audio/mp3;base64,{audio_b64}" type="audio/mp3"></audio>',
            unsafe_allow_html=True
        )
    except Exception as e:
        st.warning(f"Audio no disponible: {e}")
 
# =========================================================
# ESTADOS DE LA CASA
# =========================================================
 
defaults = {
    "door_open": False,
    "kitchen_light": False,
    "recognized_person": "",
    "last_command": "",
    "model_loaded": False,
    "model_backend": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
 
# =========================================================
# ESTILOS CSS — diseño nuevo, limpio y moderno
# =========================================================
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
 
* { box-sizing: border-box; }
 
.stApp {
    background: #080C14;
    color: #E8EDF5;
    font-family: 'DM Sans', sans-serif;
}
 
/* Ocultar elementos innecesarios de Streamlit */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 3rem 3rem; max-width: 1200px; }
 
/* ── HEADER ── */
.app-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1.5rem 2rem;
    background: linear-gradient(135deg, #0D1B2A 0%, #112240 100%);
    border-radius: 20px;
    border: 1px solid #1E3A5F;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.app-title {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #E8EDF5;
    letter-spacing: -0.5px;
    margin: 0;
}
.app-title span { color: #38BDF8; }
.app-subtitle {
    color: #64748B;
    font-size: 0.85rem;
    font-weight: 300;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin: 0;
}
.status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 6px 14px;
    border-radius: 100px;
    font-size: 0.78rem;
    font-weight: 500;
    background: #0F2035;
    border: 1px solid #1E3A5F;
    color: #94A3B8;
}
.status-pill .dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: #22C55E;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.3; }
}
 
/* ── TARJETAS DE ESTADO ── */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #38BDF8;
    margin-bottom: 1rem;
}
.state-card {
    border-radius: 18px;
    padding: 2rem 1.5rem;
    text-align: center;
    border: 1px solid;
    transition: all 0.3s ease;
    position: relative;
    overflow: hidden;
}
.state-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
}
.card-off {
    background: #0D1B2A;
    border-color: #1E293B;
}
.card-off::before { background: #1E293B; }
 
.card-door-open {
    background: linear-gradient(135deg, #052e16 0%, #064e3b 100%);
    border-color: #059669;
    box-shadow: 0 0 30px rgba(5,150,105,0.2);
}
.card-door-open::before { background: linear-gradient(90deg, #059669, #34D399); }
 
.card-door-closed {
    background: linear-gradient(135deg, #1a0a0a 0%, #2d1111 100%);
    border-color: #991B1B;
    box-shadow: 0 0 30px rgba(153,27,27,0.15);
}
.card-door-closed::before { background: linear-gradient(90deg, #991B1B, #EF4444); }
 
.card-light-on {
    background: linear-gradient(135deg, #1c1205 0%, #2d1f05 100%);
    border-color: #D97706;
    box-shadow: 0 0 30px rgba(217,119,6,0.2);
}
.card-light-on::before { background: linear-gradient(90deg, #D97706, #FCD34D); }
 
.card-icon { font-size: 2.8rem; margin-bottom: 0.8rem; display: block; }
.card-status {
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #E8EDF5;
}
.card-person {
    font-size: 0.82rem;
    color: #6EE7B7;
    margin-top: 0.4rem;
    font-weight: 500;
}
 
/* ── SECCIONES ── */
.section-card {
    background: #0D1B2A;
    border: 1px solid #1E293B;
    border-radius: 20px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
}
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1.2rem;
}
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #E8EDF5;
    margin: 0;
}
 
/* ── RESULTADO DETECCIÓN ── */
.detection-result {
    padding: 1rem 1.2rem;
    border-radius: 12px;
    margin-top: 1rem;
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 0.9rem;
    font-weight: 500;
}
.detection-ok {
    background: #052e16;
    border: 1px solid #059669;
    color: #6EE7B7;
}
.detection-fail {
    background: #1a0a0a;
    border: 1px solid #991B1B;
    color: #FCA5A5;
}
.confidence-bar {
    height: 4px;
    border-radius: 2px;
    background: #1E293B;
    margin-top: 0.8rem;
    overflow: hidden;
}
.confidence-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.5s ease;
}
 
/* ── COMANDOS DE VOZ ── */
.voice-commands-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.6rem;
    margin-top: 0.8rem;
}
.voice-cmd {
    background: #112240;
    border: 1px solid #1E3A5F;
    border-radius: 10px;
    padding: 0.6rem 0.8rem;
    font-size: 0.8rem;
    color: #94A3B8;
    display: flex;
    align-items: center;
    gap: 6px;
}
.voice-cmd span { color: #38BDF8; font-weight: 500; }
 
/* ── COMANDO DETECTADO ── */
.cmd-detected {
    background: #0c1f35;
    border: 1px solid #1E3A5F;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-top: 1rem;
    font-size: 0.9rem;
    color: #7DD3FC;
}
.cmd-detected strong { color: #38BDF8; }
 
/* ── FOOTER ── */
.app-footer {
    text-align: center;
    padding: 1rem;
    color: #334155;
    font-size: 0.75rem;
    margin-top: 1rem;
    letter-spacing: 1px;
}
 
/* Ajustes widgets Streamlit */
.stCameraInput > label { color: #64748B !important; font-size: 0.85rem !important; }
div[data-testid="stCameraInput"] { border-radius: 12px; overflow: hidden; }
 
/* Botones Streamlit */
.stButton > button {
    background: linear-gradient(135deg, #0EA5E9, #0284C7) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.5rem 1.5rem !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(14,165,233,0.3) !important;
}
</style>
""", unsafe_allow_html=True)
 
# =========================================================
# CARGAR MODELO AL INICIO
# =========================================================
 
model = None
model_error = None
 
try:
    model, backend = load_my_model()
    st.session_state.model_loaded = True
    st.session_state.model_backend = backend
except Exception as e:
    model_error = str(e)
    st.session_state.model_loaded = False
 
# =========================================================
# HEADER
# =========================================================
 
status_text = f"Modelo activo · {st.session_state.model_backend}" if st.session_state.model_loaded else "Modelo no cargado"
 
st.markdown(f"""
<div class="app-header">
    <div>
        <p class="app-title">🏠 Smart <span>Home</span> AI</p>
        <p class="app-subtitle">Sistema de acceso inteligente</p>
    </div>
    <div class="status-pill">
        <span class="dot"></span>
        {status_text}
    </div>
</div>
""", unsafe_allow_html=True)
 
# Mostrar error del modelo si existe (compacto)
if model_error:
    st.error(f"⚠️ No se pudo cargar el modelo. Verifica que `keras_model.h5` esté en la misma carpeta y que las dependencias estén instaladas correctamente.\n\n`{model_error[:200]}...`")
 
# =========================================================
# ESTADO DE LA RESIDENCIA
# =========================================================
 
st.markdown('<p class="section-label">Estado de la residencia</p>', unsafe_allow_html=True)
 
col1, col2 = st.columns(2, gap="medium")
 
with col1:
    if st.session_state.door_open:
        person = st.session_state.recognized_person
        person_html = f'<p class="card-person">👤 {person}</p>' if person else ""
        st.markdown(f"""
        <div class="state-card card-door-open">
            <span class="card-icon">🚪</span>
            <p class="card-status">Puerta Abierta</p>
            {person_html}
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="state-card card-door-closed">
            <span class="card-icon">🔒</span>
            <p class="card-status">Puerta Cerrada</p>
        </div>""", unsafe_allow_html=True)
 
with col2:
    if st.session_state.kitchen_light:
        st.markdown("""
        <div class="state-card card-light-on">
            <span class="card-icon">💡</span>
            <p class="card-status">Cocina Encendida</p>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="state-card card-off">
            <span class="card-icon">🌑</span>
            <p class="card-status">Cocina Apagada</p>
        </div>""", unsafe_allow_html=True)
 
st.markdown("<br>", unsafe_allow_html=True)
 
# =========================================================
# RECONOCIMIENTO FACIAL
# =========================================================
 
with st.container():
    st.markdown("""
    <div class="section-card">
        <div class="section-header">
            <span>📷</span>
            <p class="section-title">Reconocimiento Facial</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
 
    img_file_buffer = st.camera_input("Toma una foto para identificarte", label_visibility="collapsed")
 
    if img_file_buffer is not None:
        if model is None:
            st.warning("El modelo no está cargado. No se puede hacer reconocimiento facial.")
        else:
            img = Image.open(img_file_buffer).convert("RGB")
            img_resized = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
            img_array = np.asarray(img_resized).astype(np.float32)
            normalized = (img_array / 127.5) - 1
            data = normalized[np.newaxis, ...]  # shape (1,224,224,3)
 
            with st.spinner("Analizando..."):
                prediction = model.predict(data, verbose=0)
                index = int(np.argmax(prediction))
                confidence = float(prediction[0][index])
 
            # Barra de confianza
            bar_color = "#22C55E" if confidence > 0.80 else "#F59E0B" if confidence > 0.50 else "#EF4444"
            st.markdown(f"""
            <p style="color:#64748B; font-size:0.8rem; margin-bottom:4px;">
                Confianza de detección: <strong style="color:#E8EDF5">{confidence:.1%}</strong>
            </p>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width:{confidence*100:.0f}%; background:{bar_color};"></div>
            </div>
            """, unsafe_allow_html=True)
 
            if confidence > 0.80:
                nombres = ["María José", "Camilo"]
                persona_detectada = nombres[index] if index < len(nombres) else "Usuario"
 
                st.markdown(f"""
                <div class="detection-result detection-ok">
                    ✅ Bienvenido/a, <strong>{persona_detectada}</strong> — Acceso concedido
                </div>""", unsafe_allow_html=True)
 
                if st.session_state.recognized_person != persona_detectada:
                    st.session_state.door_open = True
                    st.session_state.recognized_person = persona_detectada
                    generar_voz(f"Bienvenida a casa {persona_detectada}. He abierto la puerta principal.")
                    st.balloons()
                    st.rerun()
            else:
                st.markdown("""
                <div class="detection-result detection-fail">
                    ❌ Rostro no reconocido — Acceso denegado
                </div>""", unsafe_allow_html=True)
 
# =========================================================
# CONTROL POR VOZ — sin bokeh, usando st.text_input nativo
# =========================================================
 
st.markdown("<br>", unsafe_allow_html=True)
 
st.markdown("""
<div class="section-card">
    <div class="section-header">
        <span>🎤</span>
        <p class="section-title">Control por Voz</p>
    </div>
</div>
""", unsafe_allow_html=True)
 
# Componente HTML nativo con Web Speech API
speech_component = """
<div style="display:flex; flex-direction:column; gap:12px;">
    <button
        id="micBtn"
        onclick="startListening()"
        style="
            background: linear-gradient(135deg, #0EA5E9, #0284C7);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 12px 24px;
            font-size: 15px;
            font-weight: 600;
            cursor: pointer;
            display: flex;
            align-items: center;
            gap: 8px;
            width: fit-content;
            transition: all 0.2s;
        "
        onmouseover="this.style.transform='translateY(-2px)'"
        onmouseout="this.style.transform='translateY(0)'"
    >
        🎤 Presiona para hablar
    </button>
    <div id="status" style="color:#64748B; font-size:0.85rem; min-height:20px;"></div>
</div>
 
<script>
function startListening() {
    const btn = document.getElementById('micBtn');
    const status = document.getElementById('status');
 
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        status.innerHTML = '<span style="color:#EF4444">⚠️ Tu navegador no soporta reconocimiento de voz. Usa Chrome.</span>';
        return;
    }
 
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const recognition = new SpeechRecognition();
    recognition.lang = 'es-ES';
    recognition.continuous = false;
    recognition.interimResults = false;
 
    btn.innerHTML = '🔴 Escuchando...';
    btn.style.background = 'linear-gradient(135deg, #DC2626, #991B1B)';
    status.innerHTML = '<span style="color:#38BDF8">Habla ahora...</span>';
 
    recognition.onresult = function(e) {
        const text = e.results[0][0].transcript;
        status.innerHTML = '<span style="color:#6EE7B7">✓ Detectado: ' + text + '</span>';
        // Pasamos el texto al input de Streamlit
        const input = window.parent.document.querySelectorAll('input[type="text"]');
        if (input.length > 0) {
            const nativeInputValueSetter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
            nativeInputValueSetter.call(input[input.length - 1], text);
            input[input.length - 1].dispatchEvent(new Event('input', { bubbles: true }));
        }
        btn.innerHTML = '🎤 Presiona para hablar';
        btn.style.background = 'linear-gradient(135deg, #0EA5E9, #0284C7)';
    };
 
    recognition.onerror = function(e) {
        status.innerHTML = '<span style="color:#EF4444">Error: ' + e.error + '</span>';
        btn.innerHTML = '🎤 Presiona para hablar';
        btn.style.background = 'linear-gradient(135deg, #0EA5E9, #0284C7)';
    };
 
    recognition.onend = function() {
        if (btn.innerHTML.includes('Escuchando')) {
            btn.innerHTML = '🎤 Presiona para hablar';
            btn.style.background = 'linear-gradient(135deg, #0EA5E9, #0284C7)';
        }
    };
 
    recognition.start();
}
</script>
"""
 
st.components.v1.html(speech_component, height=100)
 
# Input de texto para el comando (también acepta escritura manual)
col_input, col_btn = st.columns([4, 1])
with col_input:
    texto_input = st.text_input(
        "Comando de voz o escrito",
        placeholder="Ej: encender cocina, abrir puerta...",
        label_visibility="collapsed",
        key="voice_input"
    )
with col_btn:
    ejecutar = st.button("Ejecutar", use_container_width=True)
 
# Comandos disponibles
st.markdown("""
<div class="voice-commands-grid">
    <div class="voice-cmd">🍳 <span>"encender cocina"</span></div>
    <div class="voice-cmd">🌑 <span>"apagar cocina"</span></div>
    <div class="voice-cmd">🚪 <span>"abrir puerta"</span></div>
    <div class="voice-cmd">🔒 <span>"cerrar puerta"</span></div>
</div>
""", unsafe_allow_html=True)
 
# Procesar comando
if ejecutar and texto_input:
    cmd = texto_input.lower().strip()
    st.markdown(f'<div class="cmd-detected">🎙️ Comando: <strong>{texto_input}</strong></div>', unsafe_allow_html=True)
 
    if "encender cocina" in cmd:
        st.session_state.kitchen_light = True
        generar_voz("Encendiendo las luces de la cocina.")
        st.rerun()
    elif "apagar cocina" in cmd:
        st.session_state.kitchen_light = False
        generar_voz("Apagando la cocina.")
        st.rerun()
    elif "abrir puerta" in cmd:
        st.session_state.door_open = True
        generar_voz("Puerta abierta.")
        st.rerun()
    elif "cerrar puerta" in cmd:
        st.session_state.door_open = False
        st.session_state.recognized_person = ""
        generar_voz("Puerta cerrada.")
        st.rerun()
    else:
        st.warning(f"Comando no reconocido: '{texto_input}'")
 
# =========================================================
# FOOTER
# =========================================================
 
st.markdown(f"""
<div class="app-footer">
    Python {platform.python_version()} &nbsp;·&nbsp; Smart Home AI v3.0
</div>
""", unsafe_allow_html=True)
