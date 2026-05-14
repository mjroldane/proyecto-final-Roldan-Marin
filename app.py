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
# CARGA DEL MODELO
# =========================================================
 
@st.cache_resource
def load_my_model():
    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, "keras_model.h5")
    try:
        import tf_keras
        return tf_keras.models.load_model(model_path, compile=False), "tf_keras"
    except Exception:
        pass
    try:
        import tensorflow as tf
        return tf.keras.models.load_model(model_path, compile=False), "tf.keras"
    except Exception:
        pass
    try:
        from keras.models import load_model
        return load_model(model_path, compile=False), "keras"
    except Exception as e:
        raise RuntimeError(f"No se pudo cargar el modelo: {e}")
 
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
# ESTADOS
# =========================================================
 
defaults = {
    "door_open": False,
    "kitchen_light": False,
    "recognized_person": "",
    "model_loaded": False,
    "model_backend": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v
 
# =========================================================
# ESTILOS CSS
# =========================================================
 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');
 
* { box-sizing: border-box; }
.stApp { background: #080C14; color: #E8EDF5; font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 3rem 3rem; max-width: 1200px; }
 
.app-header {
    display:flex; align-items:center; justify-content:space-between;
    padding:1.5rem 2rem;
    background:linear-gradient(135deg,#0D1B2A 0%,#112240 100%);
    border-radius:20px; border:1px solid #1E3A5F;
    margin-bottom:2rem; box-shadow:0 8px 32px rgba(0,0,0,0.4);
}
.app-title { font-family:'Syne',sans-serif; font-size:2rem; font-weight:800; color:#E8EDF5; letter-spacing:-0.5px; margin:0; }
.app-title span { color:#38BDF8; }
.app-subtitle { color:#64748B; font-size:0.85rem; font-weight:300; letter-spacing:2px; text-transform:uppercase; margin:0; }
.status-pill {
    display:inline-flex; align-items:center; gap:6px;
    padding:6px 14px; border-radius:100px; font-size:0.78rem; font-weight:500;
    background:#0F2035; border:1px solid #1E3A5F; color:#94A3B8;
}
.status-pill .dot { width:7px; height:7px; border-radius:50%; background:#22C55E; animation:pulse 2s infinite; }
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }
 
.section-label { font-family:'Syne',sans-serif; font-size:0.7rem; font-weight:700; letter-spacing:3px; text-transform:uppercase; color:#38BDF8; margin-bottom:1rem; }
 
.state-card { border-radius:18px; padding:2rem 1.5rem; text-align:center; border:1px solid; position:relative; overflow:hidden; }
.state-card::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; }
.card-off { background:#0D1B2A; border-color:#1E293B; }
.card-off::before { background:#1E293B; }
.card-door-open { background:linear-gradient(135deg,#052e16,#064e3b); border-color:#059669; box-shadow:0 0 30px rgba(5,150,105,0.2); }
.card-door-open::before { background:linear-gradient(90deg,#059669,#34D399); }
.card-door-closed { background:linear-gradient(135deg,#1a0a0a,#2d1111); border-color:#991B1B; box-shadow:0 0 30px rgba(153,27,27,0.15); }
.card-door-closed::before { background:linear-gradient(90deg,#991B1B,#EF4444); }
.card-light-on { background:linear-gradient(135deg,#1c1205,#2d1f05); border-color:#D97706; box-shadow:0 0 30px rgba(217,119,6,0.2); }
.card-light-on::before { background:linear-gradient(90deg,#D97706,#FCD34D); }
.card-icon { font-size:2.8rem; margin-bottom:0.8rem; display:block; }
.card-status { font-family:'Syne',sans-serif; font-size:0.95rem; font-weight:700; letter-spacing:2px; text-transform:uppercase; color:#E8EDF5; }
.card-person { font-size:0.82rem; color:#6EE7B7; margin-top:0.4rem; font-weight:500; }
 
.section-card { background:#0D1B2A; border:1px solid #1E293B; border-radius:20px; padding:1.8rem; margin-bottom:1.5rem; }
.section-header { display:flex; align-items:center; gap:10px; margin-bottom:1.2rem; }
.section-title { font-family:'Syne',sans-serif; font-size:1.1rem; font-weight:700; color:#E8EDF5; margin:0; }
 
.detection-result { padding:1rem 1.2rem; border-radius:12px; margin-top:1rem; font-size:0.9rem; font-weight:500; }
.detection-ok { background:#052e16; border:1px solid #059669; color:#6EE7B7; }
.detection-fail { background:#1a0a0a; border:1px solid #991B1B; color:#FCA5A5; }
.confidence-bar { height:4px; border-radius:2px; background:#1E293B; margin-top:0.8rem; overflow:hidden; }
.confidence-fill { height:100%; border-radius:2px; }
 
.voice-commands-grid { display:grid; grid-template-columns:repeat(2,1fr); gap:0.6rem; margin-top:0.8rem; }
.voice-cmd { background:#112240; border:1px solid #1E3A5F; border-radius:10px; padding:0.6rem 0.8rem; font-size:0.8rem; color:#94A3B8; display:flex; align-items:center; gap:6px; }
.voice-cmd span { color:#38BDF8; font-weight:500; }
 
.cmd-detected { background:#0c1f35; border:1px solid #1E3A5F; border-radius:12px; padding:1rem 1.2rem; margin-top:1rem; font-size:0.9rem; color:#7DD3FC; }
.cmd-detected strong { color:#38BDF8; }
 
.app-footer { text-align:center; padding:1rem; color:#334155; font-size:0.75rem; margin-top:1rem; letter-spacing:1px; }
 
.stCameraInput > label { color:#64748B !important; font-size:0.85rem !important; }
div[data-testid="stCameraInput"] { border-radius:12px; overflow:hidden; }
 
.stButton > button, [data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg,#0EA5E9,#0284C7) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important;
    padding: 0.5rem 1.5rem !important; transition: all 0.2s ease !important;
}
.stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 15px rgba(14,165,233,0.3) !important;
}
[data-testid="stForm"] { border:none !important; background:transparent !important; padding:0 !important; }
</style>
""", unsafe_allow_html=True)
 
# =========================================================
# CARGAR MODELO
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
 
status_text = f"Modelo activo | {st.session_state.model_backend}" if st.session_state.model_loaded else "Modelo no cargado"
 
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
 
if model_error:
    st.error(f"⚠️ No se pudo cargar el modelo. Verifica que `keras_model.h5` esté en la misma carpeta.\n\n`{model_error[:200]}...`")
 
# =========================================================
# ESTADO DE LA RESIDENCIA + VISTA DE PÁJARO
# =========================================================
 
st.markdown('<p class="section-label">Estado de la residencia</p>', unsafe_allow_html=True)
 
col_cards, col_house = st.columns([1, 1], gap="large")
 
# ── Tarjetas de estado ──
with col_cards:
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
 
    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
 
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
 
# ── Vista de pájaro SVG ──
with col_house:
    door_color   = "#22C55E" if st.session_state.door_open else "#EF4444"
    door_label   = "ABIERTA" if st.session_state.door_open else "CERRADA"
    k_fill       = "#FDE68A" if st.session_state.kitchen_light else "#1E293B"
    k_stroke     = "#FBBF24" if st.session_state.kitchen_light else "#334155"
    k_label      = "LUZ ON"  if st.session_state.kitchen_light else "LUZ OFF"
    k_text_color = "#92400E" if st.session_state.kitchen_light else "#64748B"
    k_opacity    = "1"       if st.session_state.kitchen_light else "0.2"
    door_swing   = (
        f"<rect x='108' y='17' width='6' height='22' rx='2' fill='{door_color}' "
        f"transform='rotate(-70 108 17)' opacity='0.9'/>"
        if st.session_state.door_open else
        f"<rect x='108' y='17' width='25' height='6' rx='2' fill='{door_color}' opacity='0.9'/>"
    )
 
    st.markdown(f"""
    <div style="background:#0D1B2A; border:1px solid #1E293B; border-radius:20px; padding:1.2rem; text-align:center;">
        <p style="font-family:'Syne',sans-serif; font-size:0.7rem; font-weight:700; letter-spacing:3px;
                  text-transform:uppercase; color:#38BDF8; margin-bottom:0.8rem;">Vista de Planta</p>
        <svg viewBox="0 0 340 300" xmlns="http://www.w3.org/2000/svg"
             style="width:100%; max-width:340px; filter:drop-shadow(0 4px 20px rgba(0,0,0,0.5))">
            <defs>
                <filter id="glow">
                    <feGaussianBlur stdDeviation="5" result="b"/>
                    <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
                </filter>
                <radialGradient id="kLight" cx="50%" cy="50%" r="60%">
                    <stop offset="0%" stop-color="#FDE68A" stop-opacity="0.35"/>
                    <stop offset="100%" stop-color="#FDE68A" stop-opacity="0"/>
                </radialGradient>
            </defs>
 
            <!-- Exterior -->
            <rect x="20" y="20" width="300" height="260" rx="8"
                  fill="#0F2035" stroke="#1E3A5F" stroke-width="3"/>
 
            <!-- Habitación principal -->
            <rect x="30" y="30" width="155" height="170" rx="4"
                  fill="#112240" stroke="#1E3A5F" stroke-width="1.5"/>
            <text x="107" y="55" text-anchor="middle" fill="#334155"
                  font-size="8" font-family="sans-serif" letter-spacing="1.5">HABITACIÓN</text>
            <rect x="48" y="65" width="75" height="100" rx="6" fill="#1E293B" stroke="#334155" stroke-width="1"/>
            <rect x="48" y="65" width="75" height="18" rx="4" fill="#1E40AF" opacity="0.5"/>
            <rect x="58" y="90" width="55" height="68" rx="3" fill="#1E3A8A" opacity="0.3"/>
            <circle cx="130" cy="120" r="14" fill="#1E293B" stroke="#334155" stroke-width="1"/>
            <rect x="148" y="75" width="20" height="55" rx="3" fill="#1E293B" stroke="#334155" stroke-width="1"/>
 
            <!-- Sala -->
            <rect x="195" y="30" width="115" height="110" rx="4"
                  fill="#0F1F35" stroke="#1E3A5F" stroke-width="1.5"/>
            <text x="252" y="50" text-anchor="middle" fill="#334155"
                  font-size="8" font-family="sans-serif" letter-spacing="1.5">SALA</text>
            <rect x="204" y="60" width="97" height="32" rx="5" fill="#1E293B" stroke="#334155" stroke-width="1"/>
            <rect x="204" y="60" width="97" height="10" rx="4" fill="#334155"/>
            <rect x="204" y="60" width="9" height="32" rx="3" fill="#334155"/>
            <rect x="292" y="60" width="9" height="32" rx="3" fill="#334155"/>
            <rect x="218" y="100" width="68" height="32" rx="4" fill="#1E293B" stroke="#334155" stroke-width="1"/>
            <circle cx="252" cy="116" r="5" fill="#334155"/>
 
            <!-- Cocina — reactiva -->
            <rect x="195" y="150" width="115" height="120" rx="4"
                  fill="{k_fill}" fill-opacity="0.12"
                  stroke="{k_stroke}" stroke-width="1.5"/>
            <rect x="195" y="150" width="115" height="120" rx="4"
                  fill="url(#kLight)" opacity="{k_opacity}"/>
            <text x="252" y="172" text-anchor="middle" fill="{k_text_color}"
                  font-size="8" font-family="sans-serif" letter-spacing="1.5">COCINA</text>
            <rect x="204" y="180" width="42" height="28" rx="3"
                  fill="{k_stroke}" fill-opacity="0.3" stroke="{k_stroke}" stroke-width="1" stroke-opacity="0.5"/>
            <rect x="254" y="180" width="47" height="28" rx="3"
                  fill="{k_stroke}" fill-opacity="0.3" stroke="{k_stroke}" stroke-width="1" stroke-opacity="0.5"/>
            <circle cx="228" cy="232" r="13" fill="none" stroke="{k_stroke}" stroke-width="2" stroke-opacity="0.6"/>
            <circle cx="228" cy="232" r="6" fill="{k_stroke}" fill-opacity="0.4"/>
            <circle cx="248" cy="232" r="8" fill="none" stroke="{k_stroke}" stroke-width="2" stroke-opacity="0.4"/>
            <text x="252" y="263" text-anchor="middle" fill="{k_text_color}"
                  font-size="7" font-family="sans-serif" font-weight="bold" letter-spacing="1.5">{k_label}</text>
 
            <!-- Pasillos -->
            <rect x="185" y="30"  width="14" height="240" fill="#080C14"/>
            <rect x="30"  y="200" width="155" height="14" fill="#080C14"/>
 
            <!-- Baño -->
            <rect x="30" y="218" width="82" height="52" rx="4"
                  fill="#0F2035" stroke="#1E3A5F" stroke-width="1.5"/>
            <text x="71" y="242" text-anchor="middle" fill="#334155"
                  font-size="7" font-family="sans-serif" letter-spacing="1">BAÑO</text>
            <ellipse cx="50" cy="255" rx="9" ry="7" fill="#1E293B" stroke="#334155" stroke-width="1"/>
            <rect x="66" y="222" width="38" height="38" rx="3"
                  fill="none" stroke="#1E293B" stroke-width="1.5" stroke-dasharray="3,2"/>
 
            <!-- Lavandería -->
            <rect x="118" y="218" width="62" height="52" rx="4"
                  fill="#0F2035" stroke="#1E3A5F" stroke-width="1.5"/>
            <text x="149" y="242" text-anchor="middle" fill="#334155"
                  font-size="6.5" font-family="sans-serif" letter-spacing="0.5">LAVANDERÍA</text>
            <rect x="127" y="226" width="22" height="22" rx="11" fill="#1E293B" stroke="#334155" stroke-width="1"/>
            <circle cx="138" cy="237" r="6" fill="none" stroke="#334155" stroke-width="1"/>
            <rect x="155" y="226" width="18" height="22" rx="3" fill="#1E293B" stroke="#334155" stroke-width="1"/>
 
            <!-- Puerta principal -->
            <g filter="url(#glow)">
                <rect x="107" y="17" width="28" height="7" rx="2"
                      fill="{door_color}" opacity="0.9"/>
                {door_swing}
                <circle cx="121" cy="20" r="4" fill="{door_color}"/>
            </g>
            <text x="121" y="13" text-anchor="middle" fill="{door_color}"
                  font-size="7" font-family="sans-serif" font-weight="bold" letter-spacing="1">{door_label}</text>
 
            <!-- Brújula -->
            <text x="310" y="38" text-anchor="middle" fill="#334155" font-size="9" font-family="sans-serif">N</text>
            <polygon points="310,22 307,30 313,30" fill="#334155"/>
        </svg>
    </div>
    """, unsafe_allow_html=True)
 
st.markdown("<br>", unsafe_allow_html=True)
 
# =========================================================
# RECONOCIMIENTO FACIAL
# =========================================================
 
st.markdown("""
<div class="section-card">
    <div class="section-header">
        <span>📷</span>
        <p class="section-title">Reconocimiento Facial</p>
    </div>
</div>
""", unsafe_allow_html=True)
 
img_file_buffer = st.camera_input("foto", label_visibility="collapsed")
 
if img_file_buffer is not None:
    if model is None:
        st.warning("El modelo no está cargado.")
    else:
        img = Image.open(img_file_buffer).convert("RGB")
        img_resized = ImageOps.fit(img, (224, 224), Image.Resampling.LANCZOS)
        img_array = np.asarray(img_resized).astype(np.float32)
        normalized = (img_array / 127.5) - 1
        data = normalized[np.newaxis, ...]
 
        with st.spinner("Analizando..."):
            prediction = model.predict(data, verbose=0)
            index = int(np.argmax(prediction))
            confidence = float(prediction[0][index])
 
        bar_color = "#22C55E" if confidence > 0.80 else "#F59E0B" if confidence > 0.50 else "#EF4444"
        st.markdown(f"""
        <p style="color:#64748B; font-size:0.8rem; margin-bottom:4px;">
            Confianza: <strong style="color:#E8EDF5">{confidence:.1%}</strong>
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
# CONTROL POR VOZ
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
 
# Botón de micrófono (HTML) — inyecta texto en el text_input del form
st.components.v1.html("""
<div style="margin-bottom:6px;">
    <button id="micBtn" onclick="startListening()" style="
        background:linear-gradient(135deg,#0EA5E9,#0284C7);
        color:white; border:none; border-radius:12px;
        padding:10px 22px; font-size:14px; font-weight:600;
        cursor:pointer; transition:all 0.2s;">
        🎤 Hablar
    </button>
    <span id="micStatus" style="margin-left:12px; color:#64748B; font-size:0.82rem;"></span>
</div>
<script>
function startListening() {
    const btn = document.getElementById('micBtn');
    const status = document.getElementById('micStatus');
    if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
        status.textContent = 'Usa Chrome para reconocimiento de voz';
        status.style.color = '#EF4444'; return;
    }
    const SR = window.SpeechRecognition || window.webkitSpeechRecognition;
    const r = new SR();
    r.lang = 'es-ES'; r.continuous = false; r.interimResults = false;
    btn.textContent = '🔴 Escuchando...';
    btn.style.background = 'linear-gradient(135deg,#DC2626,#991B1B)';
    status.textContent = 'Habla ahora...'; status.style.color = '#38BDF8';
    r.onresult = function(e) {
        const text = e.results[0][0].transcript;
        status.textContent = '✓ ' + text; status.style.color = '#6EE7B7';
        const inputs = window.parent.document.querySelectorAll('input[type="text"]');
        if (inputs.length > 0) {
            const inp = inputs[inputs.length - 1];
            const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
            setter.call(inp, text);
            inp.dispatchEvent(new Event('input', {bubbles:true}));
        }
        btn.textContent = '🎤 Hablar';
        btn.style.background = 'linear-gradient(135deg,#0EA5E9,#0284C7)';
    };
    r.onerror = function(e) {
        status.textContent = 'Error: ' + e.error; status.style.color = '#EF4444';
        btn.textContent = '🎤 Hablar';
        btn.style.background = 'linear-gradient(135deg,#0EA5E9,#0284C7)';
    };
    r.onend = function() {
        if (btn.textContent.includes('Escuchando')) {
            btn.textContent = '🎤 Hablar';
            btn.style.background = 'linear-gradient(135deg,#0EA5E9,#0284C7)';
        }
    };
    r.start();
}
</script>
""", height=55)
 
# Form — garantiza que Ejecutar capture el valor del input
with st.form(key="cmd_form", clear_on_submit=True):
    col_inp, col_btn = st.columns([4, 1])
    with col_inp:
        texto_input = st.text_input(
            "cmd",
            placeholder="Ej: encender cocina, abrir puerta...",
            label_visibility="collapsed"
        )
    with col_btn:
        submitted = st.form_submit_button("Ejecutar", use_container_width=True)
 
if submitted:
    if texto_input.strip():
        cmd = texto_input.lower().strip()
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
    else:
        st.warning("Escribe o di un comando primero.")
 
st.markdown("""
<div class="voice-commands-grid">
    <div class="voice-cmd">🍳 <span>"encender cocina"</span></div>
    <div class="voice-cmd">🌑 <span>"apagar cocina"</span></div>
    <div class="voice-cmd">🚪 <span>"abrir puerta"</span></div>
    <div class="voice-cmd">🔒 <span>"cerrar puerta"</span></div>
</div>
""", unsafe_allow_html=True)
 
# =========================================================
# FOOTER
# =========================================================
 
st.markdown(f"""
<div class="app-footer">
    Python {platform.python_version()} &nbsp;|&nbsp; Smart Home AI v3.1
</div>
""", unsafe_allow_html=True)
 
