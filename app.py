import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import time

st.set_page_config(
    page_title="NeuviMedic",
    page_icon=":microscope:",
)

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('pneumonia_model.h5')
    return model

model = load_model()

def predict(image):
    img = image.resize((150, 150))  # Redimensionar la imagen a 150x150
    img = img.convert('RGB')  # Asegurarse de que la imagen tenga tres canales
    img_array = np.array(img) / 255.0  # Normalizar la imagen
    img_array = np.expand_dims(img_array, axis=0)  # Añadir una dimensión extra para el batch
    prediction = model.predict(img_array)
    return prediction



# Título del Sidebar
st.sidebar.image("logowhite.png", use_column_width=True)

st.sidebar.subheader("Menu de Navegación: ")
# Opciones del Sidebar
opcion = st.sidebar.radio("", ("Inicio", "Análisis", "Acerca de"))



# Contenido de cada página
if opcion == "Inicio":
    st.title("NeuviMedic")
    st.write("""
    ¡Bienvenido a la aplicación de Diagnóstico de Neumonía Viral con Machine Learning!
    
    Esta herramienta está diseñada para ayudar a los profesionales de la salud a diagnosticar neumonía viral de manera rápida y precisa utilizando imágenes de rayos X. Navega por las diferentes secciones utilizando la barra lateral para cargar imágenes, analizar datos, y aprender más sobre cómo funciona la aplicación y sus beneficios.
    """)
    st.subheader("Características Principales")
    st.write("""
    - **Diagnóstico Rápido:** Obtén resultados en segundos.
    - **Alta Precisión:** Utiliza modelos de machine learning entrenados con datos extensivos.
    - **Fácil de Usar:** Interfaz intuitiva para una experiencia fluida.
    """)
    st.subheader("Cómo Empezar")
    st.write("""
    1. Ve a la sección "Cargar Archivo" en la barra lateral.
    2. Sube una imagen de rayos X del paciente.
    3. Espera unos momentos para recibir el diagnóstico.
    """)
elif opcion == "Análisis":
    st.title("Análisis de Rayos X")
    st.write("Sube una imagen de rayos X para obtener un diagnóstico.")

    uploaded_file = st.file_uploader("Elige una imagen de rayos X...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Imagen cargada', use_column_width=True)
        st.write("Clasificando...")
        time.sleep(0.5)
        prediction = predict(image)
        # st.write(f"Predicción: {'El pasiente presenta neumonia viral' if prediction[0][0] > 0.5 else 'El pasiente se encuentra sin neumonia'}")

        result = 'El pasiente presenta neumonia viral' if prediction[0][0] > 0.5 else 'El pasiente se encuentra sin neumonia'
        background_color = 'orange' if result == 'Neumonía' else 'orange'
        
        # Mostrar el resultado con fondo de color
        st.markdown(f"""
            <div style="background-color: {background_color}; padding: 10px; border-radius: 5px;">
                <h3 style="color: white;">Predicción: {result}</h3>
            </div>
        """, unsafe_allow_html=True)



elif opcion == "Acerca de":
    st.header("Acerca de")
    st.write("""
    Nuestra aplicación aprovecha la potencia de las técnicas avanzadas de machine learning para detectar neumonía viral a partir de imágenes de rayos X. El propósito de esta herramienta es ofrecer un diagnóstico rápido y preciso que apoye a los profesionales de la salud en su labor diaria.
    """)
    st.write("""
    Si bien esta aplicación es una herramienta confiable para la detección de neumonía viral, es importante recordar que no reemplaza la evaluación profesional de un médico. Los resultados proporcionados por la aplicación deben ser utilizados como apoyo y no como la única base para el diagnóstico y tratamiento. Los médicos deben considerar todos los aspectos clínicos del paciente y utilizar su juicio profesional al interpretar los resultados.
    """)
    st.subheader("¿Cómo Funciona?")
    st.write("""
    1. **Carga de Imagen:** El usuario sube una imagen de rayos X del paciente a la plataforma.
    2. **Análisis de Imagen:** El modelo de machine learning analiza la imagen, identificando características clave.
    3. **Predicción:** El sistema evalúa la imagen y determina si hay signos de neumonía viral.
    4. **Resultado:** La aplicación presenta el resultado del análisis junto con una medida de confianza en la predicción.
    """)
    st.subheader("Beneficios")
    st.write("""
    - **Rápido:** Ofrece resultados en cuestión de segundos.
    - **Preciso:** Entrenado con un amplio y diverso conjunto de datos.
    - **Apoyo a los Médicos:** Ayuda a los médicos a tomar decisiones más informadas y rápidas.
    """)
    st.subheader("Instrucciones")
    st.write("""
    1. Diríjase a la sección "Cargar Archivo" en la barra lateral.
    2. Suba una imagen de rayos X.
    3. Espere unos momentos para recibir el diagnóstico.
    """)
