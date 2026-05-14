import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import os
import platform
 
# =========================================================
# CONFIGURACION DE PAGINA
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
        raise RuntimeError("No se pudo cargar el modelo: " + str(e))
 
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
 
.state-card { border-radius:18px; padding:2rem 1.5rem; text-align:center; border:1px solid; position:relative; overflow:hidden; margin-bottom:12px; }
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
 
.app-footer { text-align:center; padding:1rem; color:#334155; font-size:0.75rem; margin-top:1rem; letter-spacing:1px; }
 
.stCameraInput > label { color:#64748B !important; }
div[data-testid="stCameraInput"] { border-radius:12px; overflow:hidden; }
 
.stButton > button {
    background: linear-gradient(135deg,#0EA5E9,#0284C7) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important;
    padding: 0.5rem 1.5rem !important;
}
[data-testid="stForm"] { border:none !important; background:transparent !important; padding:0 !important; }
[data-testid="stFormSubmitButton"] > button {
    background: linear-gradient(135deg,#0EA5E9,#0284C7) !important;
    color: white !important; border: none !important; border-radius: 10px !important;
    font-weight: 600 !important; width: 100% !important;
}
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
 
if st.session_state.model_loaded:
    status_text = "Modelo activo | " + st.session_state.model_backend
else:
    status_text = "Modelo no cargado"
 
st.markdown(
    '<div class="app-header">'
    '<div>'
    '<p class="app-title">Smart <span>Home</span> AI</p>'
    '<p class="app-subtitle">Sistema de acceso inteligente</p>'
    '</div>'
    '<div class="status-pill"><span class="dot"></span>' + status_text + '</div>'
    '</div>',
    unsafe_allow_html=True
)
 
if model_error:
    st.error("No se pudo cargar el modelo. Verifica que keras_model.h5 este en la misma carpeta. " + model_error[:200])
 
# =========================================================
# ESTADO + VISTA DE PLANTA
# =========================================================
 
st.markdown('<p class="section-label">Estado de la residencia</p>', unsafe_allow_html=True)
 
col_cards, col_house = st.columns([1, 1], gap="large")
 
with col_cards:
    if st.session_state.door_open:
        person = st.session_state.recognized_person
        person_html = '<p class="card-person">Persona: ' + person + '</p>' if person else ""
        st.markdown(
            '<div class="state-card card-door-open">'
            '<span class="card-icon">&#x1F6AA;</span>'
            '<p class="card-status">Puerta Abierta</p>'
            + person_html +
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="state-card card-door-closed">'
            '<span class="card-icon">&#x1F512;</span>'
            '<p class="card-status">Puerta Cerrada</p>'
            '</div>',
            unsafe_allow_html=True
        )
 
    if st.session_state.kitchen_light:
        st.markdown(
            '<div class="state-card card-light-on">'
            '<span class="card-icon">&#x1F4A1;</span>'
            '<p class="card-status">Cocina Encendida</p>'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="state-card card-off">'
            '<span class="card-icon">&#x1F311;</span>'
            '<p class="card-status">Cocina Apagada</p>'
            '</div>',
            unsafe_allow_html=True
        )
 
with col_house:
    door_color  = "#22C55E" if st.session_state.door_open else "#EF4444"
    door_label  = "ABIERTA" if st.session_state.door_open else "CERRADA"
    k_stroke     = "#FBBF24" if st.session_state.kitchen_light else "#334155"
    k_fill_op    = "0.12"    if st.session_state.kitchen_light else "0.04"
    k_light_op   = "1"       if st.session_state.kitchen_light else "0"
    k_label      = "LUZ ON"  if st.session_state.kitchen_light else "LUZ OFF"
    k_text_color = "#D97706" if st.session_state.kitchen_light else "#475569"
 
    if st.session_state.door_open:
        door_swing = (
            "<rect x='108' y='17' width='6' height='22' rx='2' fill='" + door_color + "' "
            "transform='rotate(-70 108 17)' opacity='0.9'/>"
        )
    else:
        door_swing = (
            "<rect x='108' y='17' width='25' height='6' rx='2' fill='" + door_color + "' opacity='0.9'/>"
        )
 
    svg_html = """
<div style="background:#0D1B2A; border:1px solid #1E293B; border-radius:20px; padding:1.2rem; text-align:center; font-family:sans-serif;">
  <p style="font-size:0.7rem; font-weight:700; letter-spacing:3px; text-transform:uppercase; color:#38BDF8; margin:0 0 0.8rem 0;">VISTA DE PLANTA</p>
  <svg viewBox="0 0 340 300" xmlns="http://www.w3.org/2000/svg" style="width:100%; max-width:360px;">
    <defs>
      <filter id="glow"><feGaussianBlur stdDeviation="4" result="b"/>
        <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
      </filter>
      <radialGradient id="kLight" cx="50%" cy="50%" r="60%">
        <stop offset="0%" stop-color="#FDE68A" stop-opacity="0.4"/>
        <stop offset="100%" stop-color="#FDE68A" stop-opacity="0"/>
      </radialGradient>
    </defs>
    <rect x="20" y="20" width="300" height="260" rx="8" fill="#0F2035" stroke="#1E3A5F" stroke-width="3"/>
    <rect x="30" y="30" width="155" height="170" rx="4" fill="#112240" stroke="#1E3A5F" stroke-width="1.5"/>
    <text x="107" y="52" text-anchor="middle" fill="#334155" font-size="8" letter-spacing="1.5">HABITACION</text>
    <rect x="48" y="60" width="75" height="100" rx="6" fill="#1E293B" stroke="#334155" stroke-width="1"/>
    <rect x="48" y="60" width="75" height="18" rx="4" fill="#1E40AF" opacity="0.5"/>
    <rect x="58" y="85" width="55" height="68" rx="3" fill="#1E3A8A" opacity="0.3"/>
    <circle cx="130" cy="115" r="14" fill="#1E293B" stroke="#334155" stroke-width="1"/>
    <rect x="148" y="70" width="20" height="55" rx="3" fill="#1E293B" stroke="#334155" stroke-width="1"/>
    <rect x="195" y="30" width="115" height="110" rx="4" fill="#0F1F35" stroke="#1E3A5F" stroke-width="1.5"/>
    <text x="252" y="50" text-anchor="middle" fill="#334155" font-size="8" letter-spacing="1.5">SALA</text>
    <rect x="204" y="58" width="97" height="32" rx="5" fill="#1E293B" stroke="#334155" stroke-width="1"/>
    <rect x="204" y="58" width="97" height="10" rx="4" fill="#334155"/>
    <rect x="204" y="58" width="9" height="32" rx="3" fill="#334155"/>
    <rect x="292" y="58" width="9" height="32" rx="3" fill="#334155"/>
    <rect x="218" y="98" width="68" height="32" rx="4" fill="#1E293B" stroke="#334155" stroke-width="1"/>
    <circle cx="252" cy="114" r="5" fill="#334155"/>
    <rect x="195" y="150" width="115" height="120" rx="4"
          fill="KFILL" fill-opacity="KFILLOP" stroke="KSTROKE" stroke-width="1.5"/>
    <rect x="195" y="150" width="115" height="120" rx="4"
          fill="url(#kLight)" opacity="KLIGHTOP"/>
    <text x="252" y="172" text-anchor="middle" fill="KTEXTCOLOR" font-size="8" letter-spacing="1.5">COCINA</text>
    <rect x="204" y="180" width="42" height="28" rx="3"
          fill="KSTROKE" fill-opacity="0.25" stroke="KSTROKE" stroke-width="1" stroke-opacity="0.5"/>
    <rect x="254" y="180" width="47" height="28" rx="3"
          fill="KSTROKE" fill-opacity="0.25" stroke="KSTROKE" stroke-width="1" stroke-opacity="0.5"/>
    <circle cx="228" cy="232" r="13" fill="none" stroke="KSTROKE" stroke-width="2" stroke-opacity="0.6"/>
    <circle cx="228" cy="232" r="6" fill="KSTROKE" fill-opacity="0.35"/>
    <circle cx="248" cy="232" r="8" fill="none" stroke="KSTROKE" stroke-width="2" stroke-opacity="0.4"/>
    <text x="252" y="263" text-anchor="middle" fill="KTEXTCOLOR"
          font-size="7" font-weight="bold" letter-spacing="1.5">KLABEL</text>
    <rect x="185" y="30"  width="14" height="240" fill="#080C14"/>
    <rect x="30"  y="200" width="155" height="14" fill="#080C14"/>
    <rect x="30" y="218" width="82" height="52" rx="4" fill="#0F2035" stroke="#1E3A5F" stroke-width="1.5"/>
    <text x="71" y="242" text-anchor="middle" fill="#334155" font-size="7" letter-spacing="1">BANO</text>
    <ellipse cx="50" cy="255" rx="9" ry="7" fill="#1E293B" stroke="#334155" stroke-width="1"/>
    <rect x="66" y="222" width="38" height="38" rx="3" fill="none" stroke="#1E293B" stroke-width="1.5" stroke-dasharray="3,2"/>
    <rect x="118" y="218" width="62" height="52" rx="4" fill="#0F2035" stroke="#1E3A5F" stroke-width="1.5"/>
    <text x="149" y="240" text-anchor="middle" fill="#334155" font-size="6.5" letter-spacing="0.5">LAVANDERIA</text>
    <rect x="127" y="226" width="22" height="22" rx="11" fill="#1E293B" stroke="#334155" stroke-width="1"/>
    <circle cx="138" cy="237" r="6" fill="none" stroke="#334155" stroke-width="1"/>
    <rect x="155" y="226" width="18" height="22" rx="3" fill="#1E293B" stroke="#334155" stroke-width="1"/>
    <g filter="url(#glow)">
      <rect x="107" y="17" width="28" height="7" rx="2" fill="DOORCOLOR" opacity="0.9"/>
      DOORSWING
      <circle cx="121" cy="20" r="4" fill="DOORCOLOR"/>
    </g>
    <text x="121" y="13" text-anchor="middle" fill="DOORCOLOR"
          font-size="7" font-weight="bold" letter-spacing="1">DOORLABEL</text>
    <text x="310" y="38" text-anchor="middle" fill="#334155" font-size="9">N</text>
    <polygon points="310,22 307,30 313,30" fill="#334155"/>
  </svg>
</div>
"""
    svg_html = svg_html.replace("DOORCOLOR",   door_color)
    svg_html = svg_html.replace("DOORLABEL",   door_label)
    svg_html = svg_html.replace("DOORSWING",   door_swing)
    svg_html = svg_html.replace("KFILL",       "#FDE68A" if st.session_state.kitchen_light else "#1E293B")
    svg_html = svg_html.replace("KFILLOP",     k_fill_op)
    svg_html = svg_html.replace("KSTROKE",     k_stroke)
    svg_html = svg_html.replace("KLIGHTOP",    k_light_op)
    svg_html = svg_html.replace("KTEXTCOLOR",  k_text_color)
    svg_html = svg_html.replace("KLABEL",      k_label)
 
    st.components.v1.html(svg_html, height=380)
 
st.markdown("<br>", unsafe_allow_html=True)
 
# =========================================================
# RECONOCIMIENTO FACIAL
# =========================================================
 
st.markdown(
    '<div class="section-card">'
    '<div class="section-header"><span>&#x1F4F7;</span>'
    '<p class="section-title">Reconocimiento Facial</p>'
    '</div></div>',
    unsafe_allow_html=True
)
 
img_file_buffer = st.camera_input("foto", label_visibility="collapsed")
 
if img_file_buffer is not None:
    if model is None:
        st.warning("El modelo no esta cargado.")
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
        st.markdown(
            '<p style="color:#64748B; font-size:0.8rem; margin-bottom:4px;">'
            'Confianza: <strong style="color:#E8EDF5">' + str(round(confidence * 100, 1)) + '%</strong></p>'
            '<div class="confidence-bar">'
            '<div class="confidence-fill" style="width:' + str(round(confidence * 100)) + '%; background:' + bar_color + ';"></div>'
            '</div>',
            unsafe_allow_html=True
        )
 
        if confidence > 0.80:
            nombres = ["Maria Jose", "Camilo"]
            persona_detectada = nombres[index] if index < len(nombres) else "Usuario"
            st.markdown(
                '<div class="detection-result detection-ok">'
                'Bienvenido/a, <strong>' + persona_detectada + '</strong> - Acceso concedido'
                '</div>',
                unsafe_allow_html=True
            )
            if st.session_state.recognized_person != persona_detectada:
                st.session_state.door_open = True
                st.session_state.recognized_person = persona_detectada
                st.balloons()
                st.rerun()
        else:
            st.markdown(
                '<div class="detection-result detection-fail">'
                'Rostro no reconocido - Acceso denegado'
                '</div>',
                unsafe_allow_html=True
            )
 
# =========================================================
# PROCESAR COMANDO DE VOZ (query param → session_state)
# =========================================================
 
voz_cmd = st.query_params.get("voz", "")
if voz_cmd:
    st.query_params.clear()
    if voz_cmd == "encender_cocina":
        st.session_state.kitchen_light = True
    elif voz_cmd == "apagar_cocina":
        st.session_state.kitchen_light = False
    elif voz_cmd == "abrir_puerta":
        st.session_state.door_open = True
    elif voz_cmd == "cerrar_puerta":
        st.session_state.door_open = False
        st.session_state.recognized_person = ""
    st.rerun()
 
# =========================================================
# CONTROL POR VOZ + MANUAL
# =========================================================
 
st.markdown("<br>", unsafe_allow_html=True)
 
st.markdown(
    '<div class="section-card">'
    '<div class="section-header"><span>&#x1F3A4;</span>'
    '<p class="section-title">Control por Voz</p>'
    '</div></div>',
    unsafe_allow_html=True
)
 
# ── Bloque JS: Web Speech API (escucha) + speechSynthesis (confirma) + query param (ejecuta) ──
# Flujo: usuario habla → se detecta el comando → speechSynthesis dice la confirmación
#        → al terminar de hablar, redirige con ?voz=... → Streamlit hace rerun y aplica el cambio
st.components.v1.html("""
<style>
  #micBtn {
    background: linear-gradient(135deg, #0EA5E9, #0284C7);
    color: white; border: none; border-radius: 12px;
    padding: 12px 28px; font-size: 15px; font-weight: 600;
    cursor: pointer; transition: all 0.2s;
    display: inline-flex; align-items: center; gap: 8px;
    font-family: 'DM Sans', sans-serif;
  }
  #micBtn.listening {
    background: linear-gradient(135deg, #DC2626, #991B1B);
    animation: micPulse 1s infinite;
  }
  @keyframes micPulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(220,38,38,0.5); }
    50%       { box-shadow: 0 0 0 12px rgba(220,38,38,0); }
  }
  #micStatus {
    color: #64748B; font-size: 0.83rem;
    font-family: sans-serif; margin-top: 10px; min-height: 22px;
  }
</style>
 
<div style="display:flex; flex-direction:column; align-items:flex-start; gap:4px; padding:4px 0;">
  <button id="micBtn" onclick="startListening()">
    &#x1F3A4; Hablar
  </button>
  <span id="micStatus">Listo para escuchar</span>
</div>
 
<script>
// ─── Mapa de comandos ───────────────────────────────────────────────────────
// Cada entrada: frase detectada → { cmd: valor del query param, voz: texto que se dice }
var COMMANDS = {
  "encender cocina": { cmd: "encender_cocina", voz: "Encendiendo la luz de la cocina"  },
  "apagar cocina":   { cmd: "apagar_cocina",   voz: "Apagando la luz de la cocina"     },
  "abrir puerta":    { cmd: "abrir_puerta",    voz: "Abriendo la puerta principal"     },
  "cerrar puerta":   { cmd: "cerrar_puerta",   voz: "Cerrando la puerta principal"     }
};
 
// ─── Síntesis de voz del navegador ─────────────────────────────────────────
// Habla el texto y llama a callback() al terminar (o tras un timeout de seguridad)
function hablar(texto, callback) {
  if (!('speechSynthesis' in window)) {
    if (callback) callback();
    return;
  }
  window.speechSynthesis.cancel();
  var utter  = new SpeechSynthesisUtterance(texto);
  utter.lang  = 'es-ES';
  utter.rate  = 1.05;
  utter.pitch = 1.0;
 
  var done = false;
  function finish() {
    if (done) return;
    done = true;
    if (callback) callback();
  }
 
  utter.onend = finish;
  utter.onerror = finish;
  // Timeout de seguridad: si onend no dispara en 4 s, continua de todos modos
  setTimeout(finish, Math.max(texto.length * 70, 2000));
  window.speechSynthesis.speak(utter);
}
 
// ─── Redirige con el comando para que Python lo procese ────────────────────
function ejecutar(cmd) {
  var url = new URL(window.parent.location.href);
  url.searchParams.set('voz', cmd);
  window.parent.location.href = url.toString();
}
 
// ─── Reconocimiento de voz ─────────────────────────────────────────────────
function startListening() {
  var btn    = document.getElementById('micBtn');
  var status = document.getElementById('micStatus');
 
  if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
    status.textContent = 'Necesitas Chrome para usar el microfono';
    status.style.color = '#EF4444';
    return;
  }
 
  var SR = window.SpeechRecognition || window.webkitSpeechRecognition;
  var r  = new SR();
  r.lang           = 'es-ES';
  r.continuous     = false;
  r.interimResults = false;
 
  btn.innerHTML = '&#x23F9; Escuchando...';
  btn.classList.add('listening');
  status.textContent = 'Habla ahora...';
  status.style.color = '#38BDF8';
 
  r.onresult = function(e) {
    var text = e.results[0][0].transcript.toLowerCase().trim();
    btn.innerHTML = '&#x1F3A4; Hablar';
    btn.classList.remove('listening');
    status.textContent = 'Escuche: "' + text + '"';
    status.style.color = '#6EE7B7';
 
    // Busca el comando mas largo que coincida (prioriza frases mas especificas)
    var matched = null;
    var maxLen  = 0;
    for (var frase in COMMANDS) {
      if (text.indexOf(frase) !== -1 && frase.length > maxLen) {
        matched = frase;
        maxLen  = frase.length;
      }
    }
 
    if (matched) {
      var info = COMMANDS[matched];
      status.textContent = 'Ejecutando: ' + matched + '...';
      status.style.color = '#FCD34D';
      // Primero habla la confirmacion, despues redirige para ejecutar el cambio
      hablar(info.voz, function() {
        ejecutar(info.cmd);
      });
    } else {
      status.textContent = 'No reconoci ese comando: "' + text + '"';
      status.style.color = '#F59E0B';
      hablar('No reconoci ese comando, intenta de nuevo', null);
    }
  };
 
  r.onerror = function(e) {
    btn.innerHTML = '&#x1F3A4; Hablar';
    btn.classList.remove('listening');
    status.textContent = 'Error de microfono: ' + e.error;
    status.style.color = '#EF4444';
  };
 
  r.onend = function() {
    // Solo resetea el boton si no hubo resultado (onresult ya lo hizo)
    if (btn.classList.contains('listening')) {
      btn.innerHTML = '&#x1F3A4; Hablar';
      btn.classList.remove('listening');
    }
  };
 
  r.start();
}
</script>
""", height=110)
 
# -- Comandos disponibles --
st.markdown(
    '<p style="color:#475569; font-size:0.75rem; letter-spacing:1px; text-transform:uppercase; margin:1rem 0 0.6rem 0;">Comandos disponibles:</p>'
    '<div style="display:grid; grid-template-columns:repeat(2,1fr); gap:0.5rem; margin-bottom:1.2rem;">'
    '<div style="background:#112240; border:1px solid #1E3A5F; border-radius:10px; padding:0.5rem 0.8rem; font-size:0.8rem; color:#94A3B8;">💡 "encender cocina"</div>'
    '<div style="background:#112240; border:1px solid #1E3A5F; border-radius:10px; padding:0.5rem 0.8rem; font-size:0.8rem; color:#94A3B8;">🌑 "apagar cocina"</div>'
    '<div style="background:#112240; border:1px solid #1E3A5F; border-radius:10px; padding:0.5rem 0.8rem; font-size:0.8rem; color:#94A3B8;">🚪 "abrir puerta"</div>'
    '<div style="background:#112240; border:1px solid #1E3A5F; border-radius:10px; padding:0.5rem 0.8rem; font-size:0.8rem; color:#94A3B8;">🔒 "cerrar puerta"</div>'
    '</div>',
    unsafe_allow_html=True
)
 
# -- Botones manuales como respaldo --
st.markdown(
    '<p style="color:#475569; font-size:0.75rem; letter-spacing:1px; text-transform:uppercase; margin-bottom:0.5rem;">O usa los botones:</p>',
    unsafe_allow_html=True
)
 
col1, col2, col3, col4 = st.columns(4)
 
with col1:
    if st.button("🚪 Abrir puerta", use_container_width=True):
        st.session_state.door_open = True
        st.rerun()
with col2:
    if st.button("🔒 Cerrar puerta", use_container_width=True):
        st.session_state.door_open = False
        st.session_state.recognized_person = ""
        st.rerun()
with col3:
    if st.button("💡 Encender cocina", use_container_width=True):
        st.session_state.kitchen_light = True
        st.rerun()
with col4:
    if st.button("🌑 Apagar cocina", use_container_width=True):
        st.session_state.kitchen_light = False
        st.rerun()
 
# =========================================================
# FOOTER
# =========================================================
 
st.markdown(
    '<div class="app-footer">Python ' + platform.python_version() + ' | Smart Home AI v3.5</div>',
    unsafe_allow_html=True
)
 
