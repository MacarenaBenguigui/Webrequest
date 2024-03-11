import streamlit as st
from PyPDF2 import PdfReader
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from dotenv import load_dotenv

load_dotenv()


#con esta le paso la informacion para que la vectorice
def get_vectorstore_from_url(url):

    #get text in document format
    loader = WebBaseLoader(url)
    document = loader.load()

    #split documment
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    #create vectorstore fron the chunk
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())

    return vector_store

#vectoriza la información y la hace "cadenas" más pequeñas y legibles
def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI()

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name = "chat_hystory"),
        ("user","{input}"),
        ("user", "Teniendo en cuenta la información anterior, genera una consulta que busque información relevante a la conversación")
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever,prompt)
    return retriever_chain

#crea un documento en cadena basado en el contexto anterior, y lo combina para crear otra nueva cadena, y es capaz de dar una respuesta basandose en el contexto
def get_conversetacional_rag_chain(retriever_chain):

    llm = ChatOpenAI()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Contesta a las preguntas del usuario basándote en el siguiente contexto:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_hystory"),
        ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_response(user_input):
    #crea una cadena de conversacion
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)

    #este es el que realmente proporciona respuestas en funcion de lo que el usuario diga
    conversation_rag_chain = get_conversetacional_rag_chain(retriever_chain)


    response=conversation_rag_chain.invoke({
            "chat_hystory": st.session_state.chat_hystory,
            "input": user_query,
        })

    return response ['answer']
        



#app config
st.set_page_config (page_title= "InnevaChatBot", page_icon = ":desktop_computer:")
st.title ("InnevaChatBot:desktop_computer:")

if "chat_hystory" not in st.session_state:
    st.session_state.chat_hystory = [
        AIMessage(content = "¿Que puedo hacer por ti hoy?"),
    ]

#sidebar
with st.sidebar:
    st.header("Setting")
    website_url = st.text_input("Website URL")


#creacion de la aplicacion
if website_url is None or website_url=="":
    st.info ("Por favor, ingresa una URL valida ")
else:
    #session state
    if "chat_hystory" not in st.session_state:
        st.session_state.chat_hystory = [
            AIMessage (content="Que puedo hacer por ti hoy?")
        ]
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)

    #user input
    user_query = st.chat_input ("¿Que puedo hacer por ti hoy?")
    if user_query is not None and user_query != "":
        response= get_response(user_query)
        st.session_state.chat_hystory.append(HumanMessage(content = user_query))
        st.session_state.chat_hystory.append(AIMessage(content=response))

    #conversation
    for message in st.session_state.chat_hystory:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
