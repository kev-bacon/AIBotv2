import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

def get_response(query): 
    return "Idk man"

def get_vectorstore_from_url(url): 
    # get the text in document form 
    loader = WebBaseLoader(url)
    document = loader.load()
    
    # split into text chunks
    text_splitter = RecursiveCharacterTextSplitter() 
    document_chunks = text_splitter.split_documents(document)

    # create a vectorstore from the chunks
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())
    return vector_store


# config settings 
st.set_page_config(page_title="Chat with me!")
st.title("Chat with documents")

if "chat_history" not in st.session_state:  # does not change when you re-read application because of session state check
    st.session_state.chat_history = [
    AIMessage(content="Hello, I am a bot. How can I help you?"),
    ]

#side bar
with st.sidebar:
    st.header("Settings")
    website_url=st.text_input("Website URL")

if website_url is None or website_url == "": #handles if there is no website url
    st.info("Please enter a website URL")
else:      
    document_chunks = get_vectorstore_from_url(website_url)
    with st.sidebar:
        st.write(document_chunks)
    #handles user inputs 
    query = st.chat_input("Type your question here...")
    if query is not None and query != "":
        response = get_response(query)
        st.session_state.chat_history.append(HumanMessage(content=query))
        st.session_state.chat_history.append(AIMessage(content=response))
        print(f'query = {query} response = {response} chat_history = {st.session_state.chat_history}')

    # handles conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage): 
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
    



