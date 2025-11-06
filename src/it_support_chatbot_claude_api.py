# -*- coding: utf-8 -*-
"""IT-Support-Chatbot-CLAUDE-API.py

Adapted IT Support Chatbot using Claude API (Anthropic)
for Customer Service and Support in Healthcare

## Installation and Setup
Required packages:
pip install langchain langchain-anthropic langchain_community langchain_huggingface
pip install faiss-cpu sentence-transformers PyMuPDF anthropic
pip install streamlit python-dotenv
"""

# Core imports
from langchain_anthropic import ChatAnthropic
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.document_loaders import PyMuPDFLoader
from pathlib import Path
import os
import getpass
from dotenv import load_dotenv

# ============================================================================
# SECTION 1: ENVIRONMENT SETUP
# ============================================================================

def setup_environment():
    """
    Set up environment variables and API keys
    
    For automated/production use: Set ANTHROPIC_API_KEY in .env file or environment
    For interactive use: Will prompt if key not found
    """
    import sys
    
    load_dotenv()
    
    if "ANTHROPIC_API_KEY" not in os.environ:
        # Check if running in interactive mode
        if sys.stdin.isatty():
            # Interactive terminal - prompt for key
            os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Enter your Anthropic API Key: ")
        else:
            # Non-interactive (pipeline/automated) - fail with clear message
            raise ValueError(
                "ANTHROPIC_API_KEY not found!\n"
                "For automated/production use, set the key in one of these ways:\n"
                "1. Create .env file with: ANTHROPIC_API_KEY=your-key-here\n"
                "2. Set environment variable: export ANTHROPIC_API_KEY=your-key-here\n"
                "3. Docker: Add to docker-compose.yml environment section"
            )
    
    print("✓ Environment configured successfully")


# ============================================================================
# SECTION 2: LLM CONFIGURATION
# ============================================================================

def load_llm(model_name="claude-sonnet-4-20250514", temperature=0.7, max_tokens=4096):
    """
    Initialize Claude LLM with specified parameters
    
    Args:
        model_name: Claude model to use (default: claude-sonnet-4-20250514)
                   Available models:
                   - claude-sonnet-4-20250514 (recommended for most use cases)
                   - claude-opus-4-20250514 (highest capability)
                   - claude-sonnet-3-5-20241022 (previous generation)
        temperature: Controls randomness (0.0 = focused, 1.0 = creative)
        max_tokens: Maximum tokens in response
    
    Returns:
        ChatAnthropic: Configured Claude LLM instance
    """
    llm = ChatAnthropic(
        model=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=None,
        max_retries=2
    )
    print(f"✓ Claude LLM loaded: {model_name}")
    return llm


# ============================================================================
# SECTION 3: DOCUMENT PROCESSING
# ============================================================================

def extract_text_pdf(file_path):
    """
    Extract text content from PDF file
    
    Args:
        file_path: Path to PDF file
    
    Returns:
        str: Extracted text content
    """
    loader = PyMuPDFLoader(file_path)
    doc = loader.load()
    content = "\n".join([page.page_content for page in doc])
    return content


def load_documents(folder_path="./documents"):
    """
    Load all PDF documents from specified folder
    
    Args:
        folder_path: Path to folder containing PDF documents
    
    Returns:
        list: List of loaded document texts
    """
    docs_path = Path(folder_path)
    pdf_files = [f for f in docs_path.glob("*.pdf")]
    
    if not pdf_files:
        print(f"⚠️  No PDF files found in {folder_path}")
        return []
    
    loaded_documents = [extract_text_pdf(pdf) for pdf in pdf_files]
    print(f"✓ Loaded {len(loaded_documents)} PDF documents")
    return loaded_documents


# ============================================================================
# SECTION 4: TEXT SPLITTING & CHUNKING
# ============================================================================

def split_documents(documents, chunk_size=1000, chunk_overlap=200):
    """
    Split documents into smaller chunks for processing
    
    Args:
        documents: List of document texts
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
    
    Returns:
        list: List of text chunks
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    chunks = []
    for doc in documents:
        chunks.extend(text_splitter.split_text(doc))
    
    print(f"✓ Created {len(chunks)} text chunks")
    return chunks


# ============================================================================
# SECTION 5: EMBEDDINGS & VECTOR STORE
# ============================================================================

def create_vectorstore(chunks, embedding_model="BAAI/bge-large-en-v1.5", save_path="index_faiss"):
    """
    Create FAISS vector store from text chunks
    
    Args:
        chunks: List of text chunks
        embedding_model: HuggingFace embedding model to use
        save_path: Path to save FAISS index
    
    Returns:
        FAISS: Vector store instance
    """
    print(f"Creating embeddings with model: {embedding_model}")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    
    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    vectorstore.save_local(save_path)
    print(f"✓ Vector store created and saved to {save_path}")
    
    return vectorstore


def load_vectorstore(save_path="index_faiss", embedding_model="BAAI/bge-large-en-v1.5"):
    """
    Load existing FAISS vector store
    
    Args:
        save_path: Path to saved FAISS index
        embedding_model: HuggingFace embedding model used
    
    Returns:
        FAISS: Loaded vector store instance
    """
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    vectorstore = FAISS.load_local(save_path, embeddings, allow_dangerous_deserialization=True)
    print(f"✓ Vector store loaded from {save_path}")
    return vectorstore


# ============================================================================
# SECTION 6: RETRIEVER CONFIGURATION
# ============================================================================

def config_retriever(folder_path="./documents", force_rebuild=False):
    """
    Configure retriever with document indexing
    
    Args:
        folder_path: Path to folder containing documents
        force_rebuild: Force rebuild of vector store even if exists
    
    Returns:
        Retriever: Configured retriever instance
    """
    index_path = "index_faiss"
    
    # Check if index exists and can be loaded
    if not force_rebuild and Path(index_path).exists():
        try:
            vectorstore = load_vectorstore(index_path)
        except Exception as e:
            print(f"⚠️  Could not load existing index: {e}")
            print("Rebuilding index...")
            force_rebuild = True
    
    # Build new index if needed
    if force_rebuild or not Path(index_path).exists():
        documents = load_documents(folder_path)
        if not documents:
            raise ValueError(f"No documents found in {folder_path}")
        
        chunks = split_documents(documents)
        vectorstore = create_vectorstore(chunks, save_path=index_path)
    
    # Configure retriever with MMR search
    retriever = vectorstore.as_retriever(
        search_type='mmr',  # Maximum Marginal Relevance for diversity
        search_kwargs={'k': 3, 'fetch_k': 4}
    )
    
    print("✓ Retriever configured successfully")
    return retriever


# ============================================================================
# SECTION 7: RAG CHAIN CONFIGURATION
# ============================================================================

def config_rag_chain(llm, retriever):
    """
    Configure RAG (Retrieval Augmented Generation) chain
    
    Args:
        llm: Claude LLM instance
        retriever: Document retriever instance
    
    Returns:
        Chain: Configured RAG chain
    """
    # Contextualization prompt for chat history
    context_q_system_prompt = """Given the following chat history and the follow-up question 
    which might reference context in the chat history, formulate a standalone question which 
    can be understood without the chat history. Do NOT answer the question, just reformulate 
    it if needed and otherwise return it as is."""
    
    context_q_prompt = ChatPromptTemplate.from_messages([
        ("system", context_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "Question: {input}"),
    ])
    
    # Chain for contextualization
    history_aware_retriever = create_history_aware_retriever(
        llm=llm, 
        retriever=retriever, 
        prompt=context_q_prompt
    )
    
    # Q&A system prompt
    system_prompt = """You are a helpful IT Support virtual assistant for a healthcare organization.
    You provide accurate, concise answers to IT support questions.
    
    Use the following context to answer questions:
    - If the answer is in the context, provide a clear and helpful response
    - If you don't know the answer, say so honestly and suggest contacting IT support directly
    - Keep your answers professional, concise, and actionable
    - For healthcare IT systems, prioritize security and compliance in your guidance
    
    Context: {context}
    """
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "Question: {input}"),
    ])
    
    # Configure Q&A chain
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    # Create final RAG chain
    rag_chain = create_retrieval_chain(
        history_aware_retriever,
        qa_chain,
    )
    
    print("✓ RAG chain configured successfully")
    return rag_chain


# ============================================================================
# SECTION 8: CHAT FUNCTIONALITY
# ============================================================================

def chat_llm(rag_chain, user_input, chat_history):
    """
    Process user input through RAG chain and update chat history
    
    Args:
        rag_chain: Configured RAG chain
        user_input: User's question/message
        chat_history: List of previous messages
    
    Returns:
        tuple: (response_text, updated_chat_history)
    """
    # Add user message to history
    chat_history.append(HumanMessage(content=user_input))
    
    # Get response from RAG chain
    response = rag_chain.invoke({
        "input": user_input,
        "chat_history": chat_history
    })
    
    # Extract answer
    res = response["answer"]
    
    # Add assistant response to history
    chat_history.append(AIMessage(content=res))
    
    return res, chat_history


# ============================================================================
# SECTION 9: SIMPLE CONSOLE INTERFACE
# ============================================================================

def run_console_chat():
    """
    Run a simple console-based chat interface
    """
    print("\n" + "="*60)
    print("IT SUPPORT CHATBOT - HEALTHCARE (Claude API)")
    print("="*60 + "\n")
    
    # Setup
    setup_environment()
    
    # Initialize LLM
    llm = load_llm(
        model_name="claude-sonnet-4-20250514",
        temperature=0.7
    )
    
    # Configure retriever
    print("\nIndexing documents...")
    try:
        retriever = config_retriever(folder_path="./documents")
    except Exception as e:
        print(f"⚠️  Error setting up retriever: {e}")
        print("Make sure you have PDF documents in the './documents' folder")
        return
    
    # Configure RAG chain
    rag_chain = config_rag_chain(llm, retriever)
    
    # Initialize chat history
    chat_history = [
        AIMessage(content="Hi! I'm your IT Support assistant. How can I help you today?")
    ]
    
    print("\n" + "-"*60)
    print("Chat started! Type 'quit', 'exit', or 'bye' to end the conversation.")
    print("-"*60 + "\n")
    print(f"Assistant: {chat_history[0].content}\n")
    
    # Chat loop
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("\nAssistant: Goodbye! Feel free to reach out if you need more help.")
                break
            
            # Get response
            response, chat_history = chat_llm(rag_chain, user_input, chat_history)
            print(f"\nAssistant: {response}\n")
            
        except KeyboardInterrupt:
            print("\n\nChat interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n⚠️  Error: {e}\n")


# ============================================================================
# SECTION 10: BASIC TESTING FUNCTION
# ============================================================================

def test_basic_chat():
    """
    Test basic chat functionality with predefined questions
    """
    print("\n" + "="*60)
    print("TESTING IT SUPPORT CHATBOT")
    print("="*60 + "\n")
    
    setup_environment()
    llm = load_llm()
    
    # Test simple prompt without RAG
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful IT support assistant."),
        ("human", "{prompt}"),
    ])
    
    chain = template | llm | StrOutputParser()
    
    test_prompt = "How do I reset my password?"
    print(f"Test Question: {test_prompt}")
    print("-" * 60)
    
    response = chain.invoke({"prompt": test_prompt})
    print(f"Response: {response}\n")
    print("✓ Basic test completed successfully")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "test":
            test_basic_chat()
        elif sys.argv[1] == "console":
            run_console_chat()
        else:
            print("Usage: python it_support_chatbot_claude_api.py [test|console]")
    else:
        # Default: run console chat
        run_console_chat()