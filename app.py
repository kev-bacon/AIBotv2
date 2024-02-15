import streamlit as st
import pypdfium2 as pdfium
import os
import tempfile
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS 
from langchain_community.document_loaders import PyPDFium2Loader
from langchain_community.document_loaders import PyPDFDirectoryLoader
load_dotenv() 
# config settings 
st.set_page_config(page_title="Chat with me!")
st.title("Chat with documents")


# DEBUGGING COLOURS
class Colors:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    ENDC = '\033[0m'

def color_text(text, color):
    return f"{color}{text}{Colors.ENDC}"

#handles tmp file storage and returns file path to it 
def save_uploadedfile(uploadedfile, directory):
    temp_pdf_path = os.path.join(directory, uploadedfile.name)
    with open(temp_pdf_path, 'wb') as f:
        f.write(uploadedfile.getvalue())
    return temp_pdf_path

def get_vectorstore_from_url(url): 
    loader = WebBaseLoader(url)
    document = loader.load()
    text_splitter = RecursiveCharacterTextSplitter() 
    document_chunks = text_splitter.split_documents(document)
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(document_chunks, embeddings)
    print(f"{color_text('-------------\n URL VectorStore \n-------------\n', Colors.RED)} Document = \n {color_text(document, Colors.BLUE)}\n")
    for i, chunk in enumerate(document_chunks): 
        print(f"chunk[{i}] = {chunk}\n-------------------------------------------------------------------------\n")    
    return vector_store

def get_vectorstore_from_pdf(pdf_docs): 
    # loader =PyPDFium2Loader(pdf_docs)
    loader = PyPDFDirectoryLoader(pdf_docs)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter() 
    document_chunks = text_splitter.split_documents(documents)
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(document_chunks, embeddings)
    print(f"{color_text('-------------\n PDF VectorStore. \n-------------\n', Colors.RED)} Document = \n {color_text(documents, Colors.BLUE)}\n")
    for i, chunk in enumerate(document_chunks): 
        print(f"chunk[{i}] = {chunk}\n-------------------------------------------------------------------------\n")  
    return vector_store



## Looks at VectorDB and finds relevant information based on user question and chat history. Change prompting here, look into if it's user or human. 
def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    retriever = vector_store.as_retriever() 
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"), #this value is automatically replaced with chat history if it exist and will update as chat_history changes
        ("user", "{input}"),     #change user to human? 
        ("user", "Given the above conversation, generate a response to answer this question")    #CHANGE: System prompt can be changed here
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)        #REVISIT: Meant to find relevant documents and retrieve it Look into documentation on this 
    #DEBUGGING
    header = '\n-------------CONTEXT RETRIEVER-------------\n'
    chat_history_placeholder = MessagesPlaceholder(variable_name='chat_history')
    footer = '\n-------------------------------------------------------------------------\n'
    print(f"{header} MessagesPlaceholder(variable_name=chat_history)= \n{prompt} {footer}")

    return retriever_chain 

    
   
def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"), 
        MessagesPlaceholder(variable_name="chat_history"), 
        ("user", "{input}")
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    print(f'\n------------- CONVERSATIONAL RAG -------------\n Prompt = {prompt}\n\n Stuff_documents_chain = {stuff_documents_chain} \n\n ---------------------------------------------')
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(query): 
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store) 
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain) 
    response = conversation_rag_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input" : query
        })
    print(f"{color_text('get_response', Colors.RED)}\n response = {color_text(response, Colors.BLUE)} \n\n {color_text('---------------------------------------------', Colors.YELLOW)}")
    return response['answer']




#side bar
with st.sidebar:
    st.header("Settings")
    website_url=st.text_input("Website URL")
    pdf_docs = st.file_uploader("Upload your PDFs here and click 'Process'", accept_multiple_files=True, type="pdf")
    print(f"\n pdf docs = {pdf_docs} \n")
if (website_url is None or website_url == "") and (not pdf_docs): #handles if there is no website url
    st.info("Please enter a website URL or upload a document")
else:      
    if "chat_history" not in st.session_state:  # does not change when you re-read application because of session state check
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]
    if "vector_store" not in st.session_state: 
        if not ((website_url is None or website_url == "")):
            st.session_state.vector_store = get_vectorstore_from_url(website_url)
        else: 
            with tempfile.TemporaryDirectory() as temp_dir:
                #create a temp file inside temp_dir for each pdf_doc
                # file_path = ""
                for uploaded_file in pdf_docs:
                    save_uploadedfile(uploaded_file, temp_dir)
                st.session_state.vector_store = get_vectorstore_from_pdf(temp_dir)
                # st.session_state.vector_store = get_vectorstore_from_pdf(file_path)
                    
            


    #handles user inputs 
    query = st.chat_input("Type your question here...")
    if query is not None and query != "":
        response = get_response(query)
        st.session_state.chat_history.append(HumanMessage(content=query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # handles conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage): 
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
    



