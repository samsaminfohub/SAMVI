"""
IT Support Chatbot - Streamlit Web Application
Using Claude API (Anthropic) for Healthcare IT Support

FIXED VERSION - Uses LCEL instead of legacy chains
Run with: streamlit run app_claude_fixed.py
"""

import streamlit as st
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="IT Support Chatbot 🏥",
    page_icon="🏥",
    layout="wide"
)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def load_llm(model_name="claude-sonnet-4-20250514", temperature=0.7):
    """Load and cache Claude LLM"""
    llm = ChatAnthropic(
        model=model_name,
        temperature=temperature,
        max_tokens=4096,
        timeout=None,
        max_retries=2
    )
    return llm


def extract_text_pdf(file_path):
    """Extract text from PDF file"""
    loader = PyMuPDFLoader(file_path)
    doc = loader.load()
    content = "\n".join([page.page_content for page in doc])
    return content


@st.cache_resource
def config_retriever(folder_path="../documents"):
    """Configure retriever with document indexing"""
    
    with st.spinner("Loading and indexing documents..."):
        # Load PDF files
        docs_path = Path(folder_path)
        pdf_files = [f for f in docs_path.glob("*.pdf")]
        
        if not pdf_files:
            st.error(f"No PDF files found in {folder_path}")
            return None
        
        loaded_documents = [extract_text_pdf(pdf) for pdf in pdf_files]
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = []
        for doc in loaded_documents:
            chunks.extend(text_splitter.split_text(doc))
        
        # Create embeddings
        embedding_model = "BAAI/bge-large-en-v1.5"
        embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        
        # Create vector store
        vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
        vectorstore.save_local('index_faiss')
        
        # Configure retriever
        retriever = vectorstore.as_retriever(
            search_type='mmr',
            search_kwargs={'k': 3, 'fetch_k': 4}
        )
        
        st.success(f"✓ Indexed {len(pdf_files)} documents with {len(chunks)} chunks")
        
    return retriever


def format_docs(docs):
    """Format retrieved documents into a string"""
    return "\n\n".join([doc.page_content for doc in docs])


def config_rag_chain(llm, retriever):
    """Configure RAG chain using LCEL (LangChain Expression Language)"""
    
    # Contextualization prompt - reformulates question based on chat history
    contextualize_q_system_prompt = """Given a chat history and the latest user question 
    which might reference context in the chat history, formulate a standalone question 
    which can be understood without the chat history. Do NOT answer the question, 
    just reformulate it if needed and otherwise return it as is."""
    
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Q&A system prompt
    qa_system_prompt = """You are a helpful IT Support virtual assistant for a healthcare organization.
    You provide accurate, concise answers to IT support questions.
    
    Use the following context to answer questions:
    - If the answer is in the context, provide a clear and helpful response
    - If you don't know the answer, say so honestly and suggest contacting IT support directly
    - Keep your answers professional, concise, and actionable
    - For healthcare IT systems, prioritize security and compliance in your guidance
    
    Context: {context}
    """
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Create contextualized question chain
    contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()
    
    # Function to get contextualized question or original input
    def get_contextualized_question(input_dict):
        if input_dict.get("chat_history"):
            return contextualize_q_chain.invoke(input_dict)
        return input_dict["input"]
    
    # Create the RAG chain using LCEL
    rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(
                retriever.invoke(get_contextualized_question(x))
            )
        )
        | qa_prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain


def chat_llm(rag_chain, user_input):
    """Process user input and update chat history"""
    
    # Get response
    response = rag_chain.invoke({
        "input": user_input,
        "chat_history": st.session_state.chat_history
    })
    
    # Update chat history
    st.session_state.chat_history.append(HumanMessage(content=user_input))
    st.session_state.chat_history.append(AIMessage(content=response))
    
    return response


# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    # Title and description
    st.title("🏥 Healthcare IT Support Chatbot")
    st.markdown("*Powered by Claude AI (Anthropic)*")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_options = {
            "Claude Sonnet 4 (Recommended)": "claude-sonnet-4-20250514",
            "Claude Opus 4 (Most Capable)": "claude-opus-4-20250514",
            "Claude Sonnet 3.5": "claude-3-5-sonnet-20241022"
        }
        
        selected_model = st.selectbox(
            "Select Claude Model:",
            options=list(model_options.keys()),
            index=0
        )
        
        temperature = st.slider(
            "Temperature (Creativity):",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="Lower = more focused, Higher = more creative"
        )
        
        st.divider()
        
        # API Key check
        if not os.getenv("ANTHROPIC_API_KEY"):
            st.warning("⚠️ ANTHROPIC_API_KEY not found in environment")
            api_key = st.text_input("Enter your Anthropic API Key:", type="password")
            if api_key:
                os.environ["ANTHROPIC_API_KEY"] = api_key
        else:
            st.success("✓ API Key configured")
        
        st.divider()
        
        # Document folder
        doc_folder = st.text_input("Documents Folder:", value="./documents")
        
        if st.button("🔄 Reload Documents"):
            st.cache_resource.clear()
            st.rerun()
        
        st.divider()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
        
        st.divider()
        
        # Information
        st.markdown("### 📚 About")
        st.markdown("""
        This chatbot uses:
        - **Claude AI** for intelligent responses
        - **RAG** (Retrieval Augmented Generation)
        - **FAISS** for vector search
        - **HuggingFace** embeddings
        - **LCEL** (LangChain Expression Language)
        """)
    
    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    
    if "llm" not in st.session_state:
        st.session_state.llm = None
    
    # Load LLM if not loaded or model changed
    current_model = model_options[selected_model]
    if st.session_state.llm is None or st.session_state.get("current_model") != current_model:
        with st.spinner("Loading Claude model..."):
            st.session_state.llm = load_llm(current_model, temperature)
            st.session_state.current_model = current_model
    
    # Display welcome message if no chat history
    if not st.session_state.chat_history:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown("Hi! I'm your IT Support assistant. How can I help you today?")
    
    # Display chat history
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("user", avatar="👤"):
                st.markdown(message.content)
    
    # Chat input
    user_input = st.chat_input("Type your question here...")
    
    if user_input:
        # Display user message
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)
        
        # Get response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Thinking..."):
                try:
                    # Load retriever if not loaded
                    if st.session_state.retriever is None:
                        st.session_state.retriever = config_retriever(doc_folder)
                    
                    if st.session_state.retriever is None:
                        st.error("Failed to load documents. Please check your documents folder.")
                        return
                    
                    # Configure RAG chain
                    rag_chain = config_rag_chain(st.session_state.llm, st.session_state.retriever)
                    
                    # Get response
                    response = chat_llm(rag_chain, user_input)
                    st.markdown(response)
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    st.info("Please make sure you have:")
                    st.markdown("""
                    1. Set your ANTHROPIC_API_KEY
                    2. Added PDF documents to the documents folder
                    3. Installed all required packages:
                       ```
                       pip install langchain langchain-anthropic langchain-community 
                       pip install langchain-huggingface langchain-text-splitters
                       pip install streamlit faiss-cpu pymupdf python-dotenv sentence-transformers
                       ```
                    """)


if __name__ == "__main__":
    main()
