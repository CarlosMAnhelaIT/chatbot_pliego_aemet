import streamlit as st
import google.generativeai as genai
from pinecone import Pinecone
import re
import time

# ---------------- Configuración -----------------
GENAI_API_KEY     = st.secrets["GENAI_API_KEY"]
PINECONE_API_KEY  = st.secrets["PINECONE_API_KEY"]
INDEX_NAME = "proyecto-aemet"
MIN_SIMILARITY_SCORE = 0.50  # umbral mínimo de similitud
TOP_K = 5  # número de fragmentos a recuperar

# Inicializar APIs
genai.configure(api_key=GENAI_API_KEY)
model = genai.GenerativeModel("gemini-2.0-flash-lite")
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(INDEX_NAME)

# ---------------- Estilos -----------------
st.set_page_config(
    page_title="Asistente Pliego AEMET", 
    page_icon="☁️", 
    layout="wide"
)
st.markdown("""
    <style>
        body { font-family: 'Inter', sans-serif; color: #333; background-color: #f8f9fa; }
        .fragment-container { margin-top: 10px; border-left: 3px solid #0066cc; padding-left: 15px; background-color: #f0f5ff; border-radius: 5px; margin-bottom: 10px; padding: 15px; }
        .fragment-source { font-weight: 600; color: #0066cc; font-size: 0.85rem; margin-bottom: 5px; }
        .fragment-score { font-size: 0.85rem; color: #555; margin-bottom: 6px; }
        .fragment-content { font-size: 0.9rem; color: #333; white-space: pre-wrap; }
        .stTextInput input { border-radius: 4px; border: 1px solid #ddd; }
        h1, h2, h3 { font-weight: 600; color: #0066cc; }
    </style>
""", unsafe_allow_html=True)

# ---------------- Prompt Base -----------------
CUSTOM_PROMPT = '''
Eres un asistente especializado en resolver dudas sobre el pliego de condiciones del proyecto de Crowdsourcing para la recogida, tratamiento y difusión del impacto meteorológico de la Agencia Estatal de Meteorología (AEMET).
Tu objetivo es proporcionar respuestas precisas y claras basadas únicamente en el contenido del pliego.

Recibirás:
1. El historial de la conversación actual
2. La última consulta del usuario
3. Fragmentos relevantes extraídos del pliego

Instrucciones:
- Utiliza el historial para entender el contexto
- Responde con rigurosidad técnica, referenciando secciones y apartados del pliego
- Si la información no se encuentra en los fragmentos, indícalo claramente
- Sé conciso y estructurado en tus respuestas
- Cita textualmente partes del pliego cuando sea posible
'''

# ---------------- Estado -----------------
if "conversation" not in st.session_state:
    st.session_state.conversation = []

# ---------------- Funciones ----------------
def display_fragments(fragments):
    if not fragments:
        st.info("No se encontraron fragmentos relevantes del pliego.")
        return
    for fragment in fragments:
        st.markdown(f"""
        <div class=\"fragment-container\">
            <div class=\"fragment-source\">📄 Sección: {fragment['documento']}</div>
            <div class=\"fragment-score\">🔍 Similitud: {fragment['score']:.2%}</div>
            <div class=\"fragment-content\">{fragment['texto']}</div>
        </div>
        """, unsafe_allow_html=True)

def format_history(conversation):
    return "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])

def safe_generate_content(prompt, max_retries=3):
    """Función para generar contenido con manejo de errores y reintentos"""
    for attempt in range(max_retries):
        try:
            response = model.generate_content(prompt)
            if hasattr(response, 'candidates') and response.candidates:
                if hasattr(response.candidates[0].content, 'parts'):
                    return response.candidates[0].content.parts[0].text.strip()
            # Si llegamos aquí, hubo un problema con el formato de respuesta
            time.sleep(1)  # Esperar brevemente antes de reintentar
        except Exception as e:
            st.warning(f"Error en intento {attempt+1}: {str(e)}")
            time.sleep(1)  # Esperar antes de reintentar
    # Si todos los intentos fallan
    return "No se pudo generar una respuesta. Por favor, intenta reformular tu consulta."

def safe_embed_content(content, max_retries=3):
    """Función para generar embeddings con manejo de errores y reintentos"""
    for attempt in range(max_retries):
        try:
            embed = genai.embed_content(model="models/text-embedding-004", content=content)
            if 'embedding' in embed:
                return embed.get('embedding')
            time.sleep(1)
        except Exception as e:
            st.warning(f"Error al generar embedding (intento {attempt+1}): {str(e)}")
            time.sleep(1)
    return None

# ---------------- Interfaz Principal ----------------
st.title("☁️ Asistente del Pliego AEMET")
st.markdown("Consulta cualquier duda sobre el pliego del proyecto de AEMET.")

# Mostrar historial
for msg in st.session_state.conversation:
    with st.chat_message(msg['role'].lower()):
        st.markdown(msg['content'])
        if msg['role'] == 'Asistente' and 'fragments' in msg:
            with st.expander("📚 Mostrar fragmentos recuperados"):
                display_fragments(msg['fragments'])


# Entrada del usuario
user_input = st.chat_input("Escribe tu consulta sobre el pliego aquí...")
if user_input:
    st.session_state.conversation.append({'role': 'Usuario', 'content': user_input})
    with st.chat_message('usuario'):
        st.markdown(user_input)

    with st.spinner("Procesando tu consulta..."):
        try:
            # 1. Expansión de la consulta
            expand_prompt = f"Expande lingüísticamente esta consulta sobre el pliego de AEMET para mejorar su vectorización: \"{user_input}\". Tu respuesta deber únicamente la consulta expandida, nada más. Ten en cuenta que tu respuesta se vectorizará directamente para un RAG, por lo que responde únicamente con lo necesario."
            expanded_query = safe_generate_content(expand_prompt)
            print ("La consulta expandida es: ", expanded_query)
            
            # 2. Vectorizar consulta expandida
            query_vector = safe_embed_content(expanded_query)

            # 3. Recuperar fragmentos del pliego
            retrieved = []
            if query_vector:
                try:
                    query_res = index.query(vector=query_vector, top_k=TOP_K, include_metadata=True)
                    for m in query_res.get('matches', []):
                        score = m.get('score', 0)
                        if score >= MIN_SIMILARITY_SCORE:
                            meta = m.get('metadata', {})
                            retrieved.append({'texto': meta.get('texto',''), 'documento': meta.get('documento',''), 'score': score})
                except Exception as e:
                    st.error(f"Error al consultar Pinecone: {str(e)}")
            
            # 4. Síntesis de fragmentos
            synthesized_context = ""
            if retrieved:
                ctx = "\n---\n".join([f"[{f['documento']}]: {f['texto']}" for f in retrieved])
                synth_prompt = f"Dado estos fragmentos del pliego:\n{ctx}\n\nSintetiza y organiza la información en un contexto claro para el asistente. Ten en cuenta que tu respuesta servirá como fuente de datos a un LLM para responder a la siguiente consulta: {user_input}"  
                synthesized_context = safe_generate_content(synth_prompt)
                print("Los fragmentos sintetizados son: ", synthesized_context)
            else:
                synthesized_context = "No se encontraron fragmentos relevantes en el pliego para esta consulta."

            # 5. Generar respuesta final
            history = st.session_state.conversation[:-1]
            formatted = format_history(history)
            final_prompt = (
                f"{CUSTOM_PROMPT}\n\nHistorial de conversación:\n{formatted}\n\n"
                f"Contexto sintetizado:\n{synthesized_context}\n\n"
                f"Consulta actual: {user_input}"
            )
            answer = safe_generate_content(final_prompt)
            if not answer:
                answer = "Lo siento, no pude generar una respuesta. Por favor, intenta reformular tu consulta."
        
        except Exception as e:
            answer = f"Se produjo un error al procesar tu consulta: {str(e)}. Por favor, inténtalo de nuevo."
            retrieved = []

    # Guardar y mostrar respuesta
    st.session_state.conversation.append({'role':'Asistente','content': answer,'fragments': retrieved})
    with st.chat_message('asistente'):
        st.markdown(answer)
        with st.expander("📚 Mostrar fragmentos recuperados"):
            display_fragments(retrieved)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.subheader("Sobre esta herramienta")
    st.markdown("""
    Asistente especializado para resolver dudas del pliego de condiciones del proyecto AEMET.

    **Instrucciones:**
    1. Escribe tu consulta sobre el pliego
    2. Revisa la respuesta generada
    3. Consulta los fragmentos recuperados
    """)
