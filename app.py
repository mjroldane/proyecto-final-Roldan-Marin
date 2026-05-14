import streamlit as st
import numpy as np
from PIL import Image, ImageOps
from keras.models import load_model
from bokeh.models.widgets import Button
from bokeh.models import CustomJS
from streamlit_bokeh_events import streamlit_bokeh_events
import platform

# =========================================================
# CONFIGURACIÓN DE PÁGINA
# =========================================================

st.set_page_config(
    page_title="Smart Home AI",
    page_icon="🏠",
    layout="wide"
)

# =========================================================
# ESTADOS DE LA CASA
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

.stApp{
    background-color:#0F172A;
    color:white;
}

.main-title{
    text-align:center;
    font-size:50px;
    font-weight:bold;
    color:#E9D5FF;
}

.subtitle{
    text-align:center;
    color:#CBD5E1;
    margin-bottom:30px;
}

.room{
    padding:30px;
    border-radius:20px;
    text-align:center;
    margin-bottom:20px;
    box-shadow:0px 4px 20px rgba(0,0,0,0.3);
}

.room-off{
    background-color:#1E293B;
    border:2px solid #334155;
}

.room-on{
    background-color:#FDE68A;
    color:black;
    border:2px solid #FBBF24;
}

.door-open{
    background-color:#86EFAC;
    color:black;
    border:2px solid #22C55E;
}

.door-closed{
    background-color:#7F1D1D;
    border:2px solid #EF4444;
}

.camera-box{
    background-color:#111827;
    padding:20px;
    border-radius:20px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# TÍTULO
# =========================================================

st.markdown('<p class="main-title">🏠 SMART HOME AI</p>', unsafe_allow_html=True)

st.markdown(
    '<p class="subtitle">Reconocimiento Facial + Control por Voz</p>',
    unsafe_allow_html=True
)

# =========================================================
# CARGAR MODELO IA
# =========================================================

@st.cache_resource
def load_my_model():
    return load_model("keras_model.h5")

try:
    model = load_my_model()
except Exception as e:
    st.error(f"Error cargando modelo: {e}")

# =========================================================
# CASA VISUAL
# =========================================================

st.subheader("Vista Pajarito de la Casa")

col1, col2 = st.columns(2)

# =========================================================
# PUERTA
# =========================================================

with col1:

    if st.session_state.door_open:

        st.markdown(f"""
        <div class="room door-open">
            <h1>🚪</h1>
            <h2>PUERTA ABIERTA</h2>
            <h3>Bienvenida {st.session_state.recognized_person}</h3>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="room door-closed">
            <h1>🚪</h1>
            <h2>PUERTA CERRADA</h2>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# COCINA
# =========================================================

with col2:

    if st.session_state.kitchen_light:

        st.markdown("""
        <div class="room room-on">
            <h1>💡🍳</h1>
            <h2>COCINA ENCENDIDA</h2>
        </div>
        """, unsafe_allow_html=True)

    else:

        st.markdown("""
        <div class="room room-off">
            <h1>💡</h1>
            <h2>COCINA APAGADA</h2>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# RECONOCIMIENTO FACIAL
# =========================================================

st.divider()

st.header("📷 Reconocimiento Facial")

img_file_buffer = st.camera_input("Toma una foto")

if img_file_buffer is not None:

    img = Image.open(img_file_buffer).convert("RGB")

    size = (224, 224)

    img = ImageOps.fit(
        img,
        size,
        Image.Resampling.LANCZOS
    )

    img_array = np.asarray(img)

    normalized_image_array = (
        img_array.astype(np.float32) / 127.5
    ) - 1

    data = np.ndarray(
        shape=(1, 224, 224, 3),
        dtype=np.float32
    )

    data[0] = normalized_image_array

    with st.spinner("Reconociendo rostro..."):

        prediction = model.predict(data)

        index = np.argmax(prediction)

        confidence_score = prediction[0][index]

    st.write(f"Confianza: {confidence_score:.2%}")

    # =====================================================
    # PERSONA 1
    # =====================================================

    if confidence_score > 0.80:

        if index == 0:

            st.session_state.door_open = True
            st.session_state.recognized_person = "María José"

            st.success("✨ Bienvenida María José")
            st.balloons()

        elif index == 1:

            st.session_state.door_open = True
            st.session_state.recognized_person = "Camilo"

            st.success("👋 Hola Camilo")

        else:

            st.warning("Persona reconocida")

    else:

        st.error("No se pudo reconocer correctamente")

# =========================================================
# CONTROL POR VOZ
# =========================================================

st.divider()

st.header("🎤 Control por Voz")

st.write("Presiona el botón y di un comando.")

st.write("""
### Comandos disponibles:
- encender cocina
- apagar cocina
- abrir puerta
- cerrar puerta
""")

# =========================================================
# BOTÓN DE VOZ
# =========================================================

stt_button = Button(
    label="🎤 Iniciar Voz",
    width=250
)

stt_button.js_on_event(
    "button_click",
    CustomJS(code="""
        var recognition = new webkitSpeechRecognition();

        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = 'es-ES';

        recognition.onresult = function(e) {

            var value = e.results[0][0].transcript;

            document.dispatchEvent(
                new CustomEvent(
                    "GET_TEXT",
                    {detail: value}
                )
            );
        }

        recognition.start();
    """)
)

result = streamlit_bokeh_events(
    stt_button,
    events="GET_TEXT",
    key="listen",
    refresh_on_update=False,
    override_height=75,
    debounce_time=0
)

# =========================================================
# LÓGICA DE VOZ
# =========================================================

if result:

    if "GET_TEXT" in result:

        texto_detectado = result.get("GET_TEXT").lower()

        st.success(f"Detectado: {texto_detectado}")

        # =================================================
        # COMANDOS
        # =================================================

        if "encender cocina" in texto_detectado:

            st.session_state.kitchen_light = True

            st.success("💡 Cocina encendida")

        elif "apagar cocina" in texto_detectado:

            st.session_state.kitchen_light = False

            st.warning("💡 Cocina apagada")

        elif "abrir puerta" in texto_detectado:

            st.session_state.door_open = True

            st.success("🚪 Puerta abierta")

        elif "cerrar puerta" in texto_detectado:

            st.session_state.door_open = False

            st.warning("🚪 Puerta cerrada")

# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(f"Python {platform.python_version()} | Smart Home AI")
