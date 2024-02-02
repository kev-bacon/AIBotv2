import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

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


def get_vectorstore_from_url(url): 
    # get the text in document form 
    loader = WebBaseLoader(url)
    document = loader.load()
    # split into text chunks
    text_splitter = RecursiveCharacterTextSplitter() 
    document_chunks = text_splitter.split_documents(document)

    # create a vectorstore from the chunks
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())
    
    print(f"{color_text('INSIDE get_vectorstore_from_url.\n', Colors.RED)} Document = {color_text(document, Colors.BLUE)}\n Document_chunks = {color_text(document_chunks, Colors.WHITE)} \n\n {color_text('-------------------------------------------------------------------------', Colors.RED)}")
    return vector_store

## Looks at VectorDB and finds relevant information based on user question and chat history. Change prompting here, look into if it's user or human. 
def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    retriever = vector_store.as_retriever() 
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"), #this value is automatically replaced with chat history if it exist and will update as chat_history changes
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a response to answer this question")    #CHANGE: System prompt can be changed here
    ])
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)        #REVISIT: Meant to find relevant documents and retrieve it Look into documentation on this 
    print(f"{color_text('INSIDE get_context_retriever_chain.', Colors.RED)} PROMPT = {color_text(prompt, Colors.YELLOW)}\n\n Retriever_chain = {color_text(retriever_chain, Colors.CYAN)} \n\n {color_text('-------------------------------------------------------------------------', Colors.RED)}")
    return retriever_chain 

def get_conversational_rag_chain(retriever_chain):
    llm = ChatOpenAI(temperature=0, model="gpt-4")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"), 
        MessagesPlaceholder(variable_name="chat_history"), 
        ("user", "{input}")
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    print(f'INSIDE get_conversational_rag_chain. Prompt = {prompt}\n\n Stuff_documents_chain = {stuff_documents_chain} \n\n ---------------------------------------------')
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)

def get_response(query): 
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store) 
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain) 
    response = conversation_rag_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "input" : query
        })
    print(f"{color_text('INSIDE get_response', Colors.RED)}, response = {color_text(response, Colors.BLUE)} \n\n {color_text('---------------------------------------------', Colors.YELLOW)}")
    return response['answer']




#side bar
with st.sidebar:
    st.header("Settings")
    website_url=st.text_input("Website URL")

if website_url is None or website_url == "": #handles if there is no website url
    st.info("Please enter a website URL")
else:      
    if "chat_history" not in st.session_state:  # does not change when you re-read application because of session state check
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am a bot. How can I help you?"),
        ]
    if "vector_store" not in st.session_state: 
        st.session_state.vector_store = get_vectorstore_from_url(website_url)

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
    



