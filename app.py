
import streamlit as st
import pandas as pd
import joblib
from groq import Groq
import uuid

# ==========================================
# CONFIGURACION GENERAL
# ==========================================

st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="🏭",
    layout="wide"
)

# ==========================================
# ESTILOS
# ==========================================

st.markdown("""
<style>
    .stApp {
        background-color: #0b1120;
        color: #e2e8f0;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
    }

    .main-title {
        font-size: 34px;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 5px;
    }

    .subtitle {
        color: #94a3b8;
        font-size: 15px;
        margin-bottom: 25px;
    }

    .section-title {
        color: #c4b5fd;
        font-size: 20px;
        font-weight: 700;
        margin-top: 20px;
    }

    .status-normal {
        background-color: #064e3b;
        padding: 16px;
        border-radius: 12px;
        border-left: 5px solid #34d399;
    }

    .status-warning {
        background-color: #78350f;
        padding: 16px;
        border-radius: 12px;
        border-left: 5px solid #fbbf24;
    }

    .status-critical {
        background-color: #7f1d1d;
        padding: 16px;
        border-radius: 12px;
        border-left: 5px solid #f87171;
    }

    div[data-testid="stMetric"] {
        background-color: #172033;
        border: 1px solid #28364f;
        padding: 18px;
        border-radius: 12px;
    }

    .footer {
        color: #64748b;
        text-align: center;
        padding: 25px;
        font-size: 12px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CARGAR MODELO
# ==========================================

@st.cache_resource
def cargar_modelo():
    return joblib.load("modelo_mantenimiento.pkl")


try:
    modelo = cargar_modelo()
except Exception as e:
    st.error(f"No se pudo cargar el modelo: {e}")
    st.stop()

# ==========================================
# ENCABEZADO
# ==========================================

st.markdown(
    '<div class="main-title">🏭 Predictive Maintenance AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Sistema inteligente de monitoreo y mantenimiento industrial</div>',
    unsafe_allow_html=True
)

st.divider()

# ==========================================
# PANEL LATERAL: SENSORES
# ==========================================

st.sidebar.title("⚙️ Panel de sensores")
st.sidebar.markdown("Configura las lecturas de la maquinaria.")

temperatura = st.sidebar.slider(
    "🌡️ Temperatura (°C)",
    min_value=40.0,
    max_value=120.0,
    value=80.0,
    step=0.5
)

vibracion = st.sidebar.slider(
    "〰️ Vibración (mm/s)",
    min_value=0.0,
    max_value=12.0,
    value=4.5,
    step=0.1
)

horas = st.sidebar.slider(
    "⏱️ Horas de operación",
    min_value=0,
    max_value=5000,
    value=2000,
    step=50
)

presion = st.sidebar.slider(
    "🔧 Presión (PSI)",
    min_value=70.0,
    max_value=170.0,
    value=116.0,
    step=0.5
)

st.sidebar.divider()

st.sidebar.caption(
    "ISOF CLASS | Sistema Predictivo de Mantenimiento"
)

# ==========================================
# DATOS DE ENTRADA
# ==========================================

entrada = pd.DataFrame(
    [[temperatura, vibracion, horas, presion]],
    columns=[
        "temperatura_c",
        "vibracion_mms",
        "horas_operacion",
        "presion_psi"
    ]
)

# ==========================================
# PREDICCION
# ==========================================

prediccion = int(modelo.predict(entrada)[0])

probabilidades = modelo.predict_proba(entrada)[0]

clases = list(modelo.classes_)

probabilidad = float(
    probabilidades[clases.index(prediccion)]
) * 100

# ==========================================
# TARJETAS DE SENSORES
# ==========================================

st.markdown(
    '<div class="section-title">📊 Lecturas actuales de maquinaria</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Temperatura",
    f"{temperatura} °C"
)

col2.metric(
    "Vibración",
    f"{vibracion} mm/s"
)

col3.metric(
    "Operación",
    f"{horas} hrs"
)

col4.metric(
    "Presión",
    f"{presion} PSI"
)

st.divider()

# ==========================================
# DIAGNOSTICO DEL MODELO
# ==========================================

st.markdown(
    '<div class="section-title">🧠 Diagnóstico inteligente</div>',
    unsafe_allow_html=True
)

if prediccion == 0:

    st.success(
        "🟢 OPERACIÓN NORMAL: La maquinaria presenta parámetros estables."
    )

    st.markdown("""
    <div class="status-normal">
        <h3 style="color:#34d399;">Sistema estable</h3>
        <p style="color:#ffffff;">
        No se detectan condiciones que requieran mantenimiento inmediato.
        Se recomienda continuar con el monitoreo periódico.
        </p>
    </div>
    """, unsafe_allow_html=True)

elif prediccion == 1:

    st.warning(
        "🟡 MANTENIMIENTO PREVENTIVO: Se detectan señales de desgaste."
    )

    st.markdown("""
    <div class="status-warning">
        <h3 style="color:#fbbf24;">Atención requerida</h3>
        <p style="color:#ffffff;">
        La maquinaria presenta condiciones que requieren una revisión
        técnica programada para evitar posibles fallos.
        </p>
    </div>
    """, unsafe_allow_html=True)

else:

    st.error(
        "🔴 FALLO CRÍTICO: Se detectan condiciones de riesgo."
    )

    st.markdown("""
    <div class="status-critical">
        <h3 style="color:#f87171;">Alerta crítica</h3>
        <p style="color:#ffffff;">
        Se recomienda detener la maquinaria de forma segura y solicitar
        una intervención técnica inmediata.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# PROBABILIDADES
# ==========================================

st.markdown(
    '<div class="section-title">📈 Probabilidad por categoría</div>',
    unsafe_allow_html=True
)

nombres_estados = {
    0: "Operación normal",
    1: "Mantenimiento preventivo",
    2: "Fallo crítico"
}

for clase, valor in zip(clases, probabilidades):

    nombre = nombres_estados[int(clase)]

    st.write(f"**{nombre}: {valor * 100:.2f}%**")

    st.progress(float(valor))

st.info(
    f"Confianza del modelo en su predicción: {probabilidad:.2f}%"
)

# ==========================================
# CONEXION CON GROQ
# ==========================================

st.divider()

st.markdown(
    '<div class="section-title">🤖 Generador de órdenes de trabajo con IA</div>',
    unsafe_allow_html=True
)

def generar_orden():

    try:

        api_key = st.secrets["GROQ_API_KEY"]

        client = Groq(api_key=api_key)

        # Detectar modelos activos
        modelos_activos = [
            m.id for m in client.models.list().data
            if not any(
                x in m.id.lower()
                for x in [
                    "whisper",
                    "guard",
                    "tts",
                    "playai",
                    "embed"
                ]
            )
        ]

        modelos_llama = [
            m for m in modelos_activos
            if "llama" in m.lower()
        ]

        if modelos_llama:
            modelo_groq = modelos_llama[0]
        elif modelos_activos:
            modelo_groq = modelos_activos[0]
        else:
            st.error("No se encontraron modelos de chat disponibles.")
            return

        # Diagnóstico
        diagnostico = nombres_estados[prediccion]

        # Instrucciones para la IA
        system_prompt = """
        Eres el Jefe de Mantenimiento Industrial de una planta
        de manufactura automatizada.

        Tu función es analizar lecturas de sensores y generar
        diagnósticos y órdenes de trabajo técnicas en español.

        Si el estado es 1 o 2, genera una orden de trabajo
        en formato Markdown que incluya:

        1. Folio de la orden.
        2. Diagnóstico técnico.
        3. Tres acciones técnicas concretas.
        4. Equipo de Protección Personal (EPP).

        Para estado 1, genera una orden preventiva programada.

        Para estado 2, genera una orden urgente, incluyendo
        protocolo de paro seguro y aislamiento de energía.

        Nunca indiques que una acción física ya fue realizada.
        Aclara que las intervenciones deben seguir los protocolos
        de seguridad de la planta y ser realizadas por personal
        autorizado.
        """

        if prediccion == 0:

            return """
            ## Dictamen de monitoreo

            **Estado:** Operación normal.

            Los parámetros se encuentran dentro de los patrones
            normales del conjunto de entrenamiento.

            **Recomendación:** Continuar con el monitoreo periódico
            y el mantenimiento preventivo programado.
            """

        datos_sensores = f"""
        Temperatura: {temperatura} °C
        Vibración: {vibracion} mm/s
        Horas de operación: {horas}
        Presión: {presion} PSI

        Estado detectado: {prediccion}
        Diagnóstico: {diagnostico}
        Confianza del modelo: {probabilidad:.2f}%
        """

        respuesta = client.chat.completions.create(
            model=modelo_groq,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": (
                        "Genera la orden de trabajo correspondiente "
                        "a estas lecturas:\n" + datos_sensores
                    )
                }
            ],
            temperature=0.3,
            max_tokens=1200
        )

        return respuesta.choices[0].message.content

    except Exception as e:

        raise RuntimeError(
            f"No fue posible generar la respuesta de Groq: {e}"
        )


# ==========================================
# BOTON DE GENERACION
# ==========================================

if prediccion == 0:

    st.success(
        "La maquinaria está estable. No se requiere generar "
        "una orden de trabajo correctiva."
    )

else:

    if st.button(
        "⚡ Generar orden de trabajo con IA",
        use_container_width=True
    ):

        with st.spinner("Groq está analizando las lecturas..."):

            try:

                orden = generar_orden()

                st.session_state["orden_actual"] = orden

            except Exception as e:

                st.error(str(e))

                st.warning(
                    "El diagnóstico de Machine Learning continúa "
                    "disponible aunque Groq no responda."
                )

    if "orden_actual" in st.session_state:

        st.markdown("### 📋 Orden de trabajo generada")

        st.markdown(st.session_state["orden_actual"])

        st.download_button(
            "📥 Descargar orden de trabajo",
            data=st.session_state["orden_actual"],
            file_name="orden_trabajo.md",
            mime="text/markdown",
            use_container_width=True
        )

# ==========================================
# PIE DE PAGINA
# ==========================================

st.divider()

st.markdown("""
<div class="footer">
    PREDICTIVE MAINTENANCE AI | ISOF CLASS<br>
    Sistema de Machine Learning e Inteligencia Artificial Generativa
</div>
""", unsafe_allow_html=True)
