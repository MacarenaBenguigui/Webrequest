import os
from dotenv import load_dotenv
import pandas as pd
import streamlit as st
from llama_index.core.query_engine import PandasQueryEngine
from prompt import new_prompt, instruction_str, context
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core import StorageContext, VectorStoreIndex, load_index_from_storage
from llama_index.readers.file import PDFReader

load_dotenv()

#Carga de PDFs
def get_index(data, index_name):
    index = None
    if not os.path.exists(index_name):
        print("building index", index_name)
        index = VectorStoreIndex.from_documents(data, show_progress=True)
        index.storage_context.persist(persist_dir=index_name)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=index_name)
        )

    return index


pdf_path = os.path.join("data", "PORTFOLIO INNEVA.pdf")
portfolio_pdf = PDFReader().load_data(file=pdf_path)
portfolio_index = get_index(portfolio_pdf, "portfolio")
portfolio_engine = portfolio_index.as_query_engine()


load_dotenv()

#Lectura de PDFs

tools = [

     QueryEngineTool(query_engine=portfolio_engine, 
                    metadata=ToolMetadata(
                    name = "portfolio_data",
                    description = " this gives information about the portfolio of Innevapharma",

        ),
    ),
]


llm = OpenAI(model = "gpt-3.5-turbo-16k")
agent = ReActAgent.from_tools(tools, llm = llm, verbose =True, context= context)




# Interfaz de usuario para ingreso de consultas
st.set_page_config (page_title= "InnevaChatBot", page_icon = ":desktop_computer:", layout="wide")
st.title ("InnevaChatBot:desktop_computer:")
hide_streamlit_style = """
            <style>
            [data-testid="stToolbar"] {visibility: hidden !important;}
            footer {visibility: hidden !important;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
# Usa la clave 'query_input' para manejar el estado del input del usuario
user_query = st.text_input("¿En que puedo ayudarte hoy?")

if 'queries' not in st.session_state:
    st.session_state['queries'] = []

if user_query:  # Asegúrate de que el campo de consulta no esté vacío
    # Simula la obtención de una respuesta (reemplaza esto con tu lógica real de consulta)
    response_object = portfolio_engine.query(user_query)
    response_text = response_object.response
    
    # Guarda la consulta y la respuesta en el estado de sesión
    st.session_state['queries'].append({'query': user_query, 'response': response_text})
    
    # Limpia el campo de entrada para la próxima consulta
    st.session_state['query_input'] = ""  # Esto restablecerá el campo de texto

# Muestra las consultas y respuestas previas
for pair in st.session_state['queries']:
    st.markdown(f":speech_balloon: {pair['query']}")
    st.markdown(f":robot_face: {pair['response']}")
    st.write("---")  # Solo para añadir una línea divisoria por estética
