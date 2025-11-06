# IT Support Chatbot for Healthcare - Claude API Version

A production-ready IT support chatbot adapted for healthcare environments using Anthropic's Claude API and Retrieval Augmented Generation (RAG).

## 🌟 Features

- **Claude AI Integration**: Uses Anthropic's Claude models for intelligent, context-aware responses
- **RAG Implementation**: Retrieval Augmented Generation for accurate, document-based answers
- **Healthcare Focus**: Designed for healthcare IT support scenarios
- **Multi-Format Support**: Processes PDF documents for knowledge base
- **Chat History**: Maintains conversation context for better responses
- **Streamlit Interface**: User-friendly web interface
- **Vector Search**: FAISS-based semantic search for relevant information
- **Flexible Deployment**: Console or web-based interfaces

## 📋 Requirements

### Python Version
- Python 3.9 or higher

### Core Dependencies
```
langchain>=0.1.0
langchain-anthropic>=0.1.0
langchain-community>=0.0.20
langchain-huggingface>=0.0.1
anthropic>=0.18.0
faiss-cpu>=1.7.4
sentence-transformers>=2.2.2
PyMuPDF>=1.23.0
streamlit>=1.30.0
python-dotenv>=1.0.0
```

## 🚀 Installation

### 1. Clone or Download

```bash
# Create project directory
mkdir it-support-chatbot
cd it-support-chatbot
```

### 2. Install Dependencies

```bash
pip install langchain langchain-anthropic langchain-community langchain-huggingface
pip install faiss-cpu sentence-transformers PyMuPDF anthropic
pip install streamlit python-dotenv
```

Or use requirements.txt:

```bash
pip install -r requirements.txt
```

### 3. Set Up Environment

Create a `.env` file in the project root:

```bash
ANTHROPIC_API_KEY=your_api_key_here
```

To get an API key:
1. Go to https://console.anthropic.com/
2. Sign up or log in
3. Navigate to API Keys section
4. Create a new API key

### 4. Prepare Documents

Create a `documents` folder and add your PDF files:

```bash
mkdir documents
# Copy your IT support documentation PDFs to this folder
```

## 💻 Usage

### Console Interface

Run the chatbot in terminal:

```bash
python it_support_chatbot_claude_api.py console
```

### Basic Test

Test basic functionality:

```bash
python it_support_chatbot_claude_api.py test
```

### Streamlit Web Interface

Launch the web application:

```bash
streamlit run app_claude.py
```

The application will open in your browser at `http://localhost:8501`

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│           User Interface                     │
│  (Streamlit Web App / Console)              │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│         Claude LLM (Anthropic)              │
│  - claude-sonnet-4-20250514                 │
│  - claude-opus-4-20250514                   │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│           RAG Pipeline                       │
│  ┌────────────────────────────────────┐    │
│  │  1. Query Contextualization        │    │
│  │  2. History-Aware Retrieval        │    │
│  │  3. Document Retrieval (FAISS)     │    │
│  │  4. Response Generation            │    │
│  └────────────────────────────────────┘    │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│        Vector Store (FAISS)                 │
│  - HuggingFace Embeddings                   │
│  - BAAI/bge-large-en-v1.5                   │
└──────────────────┬──────────────────────────┘
                   │
┌──────────────────▼──────────────────────────┐
│      Knowledge Base (PDF Documents)         │
│  - IT Support Manuals                       │
│  - FAQ Documents                            │
│  - Troubleshooting Guides                   │
└─────────────────────────────────────────────┘
```

## 📁 Project Structure

```
it-support-chatbot/
├── it_support_chatbot_claude_api.py  # Main chatbot script
├── app_claude.py                      # Streamlit web interface
├── requirements.txt                   # Python dependencies
├── .env                              # Environment variables
├── README.md                         # This file
├── SETUP_GUIDE.md                    # Detailed setup instructions
├── documents/                        # PDF knowledge base
│   ├── manual1.pdf
│   └── manual2.pdf
└── index_faiss/                      # Vector store (auto-generated)
    ├── index.faiss
    └── index.pkl
```

## 🔧 Configuration

### Model Selection

Available Claude models:

1. **claude-sonnet-4-20250514** (Recommended)
   - Best balance of speed and capability
   - Ideal for most use cases
   - Cost-effective

2. **claude-opus-4-20250514**
   - Highest capability
   - Best for complex queries
   - Higher cost

3. **claude-sonnet-3-5-20241022**
   - Previous generation
   - Stable and reliable

### Parameters

Adjust in code or Streamlit interface:

- **Temperature**: 0.0 - 1.0 (default: 0.7)
  - Lower = more focused and deterministic
  - Higher = more creative and varied

- **Max Tokens**: Maximum response length (default: 4096)

- **Chunk Size**: Text splitting size (default: 1000)

- **Chunk Overlap**: Overlap between chunks (default: 200)

- **Retriever k**: Number of documents to retrieve (default: 3)

## 📚 Key Components

### 1. Document Processing
- **PDF Loader**: PyMuPDF for text extraction
- **Text Splitter**: Recursive character splitting for optimal chunks
- **Metadata**: Preserves document structure

### 2. Embeddings & Vector Store
- **Model**: BAAI/bge-large-en-v1.5 (HuggingFace)
- **Dimension**: 1024
- **Store**: FAISS (Facebook AI Similarity Search)
- **Search**: Maximum Marginal Relevance (MMR)

### 3. RAG Chain
- **Contextualization**: History-aware query reformulation
- **Retrieval**: Semantic search for relevant documents
- **Generation**: Claude-based response synthesis

### 4. Chat Management
- **History**: Maintains conversation context
- **Memory**: Session-based storage
- **Messages**: LangChain message format (Human/AI)

## 🎯 Use Cases

### Healthcare IT Support Scenarios

1. **Password Reset**
   - "How do I reset my password?"
   - "I forgot my login credentials"

2. **System Access**
   - "How do I access the EHR system?"
   - "VPN connection issues"

3. **Software Installation**
   - "How to install clinical software?"
   - "Printer driver installation"

4. **Troubleshooting**
   - "My email isn't working"
   - "Computer won't start"

5. **Security & Compliance**
   - "How to report a security incident?"
   - "HIPAA compliance guidelines"

## 🔒 Security Considerations

### API Key Management
- Store API keys in `.env` file
- Never commit `.env` to version control
- Add `.env` to `.gitignore`

### Healthcare Data
- Ensure HIPAA compliance
- No PHI in training documents
- Secure document storage
- Access control for knowledge base

### Deployment
- Use HTTPS in production
- Implement authentication
- Audit logging
- Regular security updates

## 🚦 Troubleshooting

### Common Issues

1. **API Key Error**
   ```
   Error: ANTHROPIC_API_KEY not found
   ```
   Solution: Set your API key in `.env` file

2. **No Documents Found**
   ```
   Warning: No PDF files found
   ```
   Solution: Add PDF files to `./documents` folder

3. **Import Errors**
   ```
   ModuleNotFoundError: No module named 'langchain_anthropic'
   ```
   Solution: Install missing packages
   ```bash
   pip install langchain-anthropic
   ```

4. **FAISS Index Error**
   ```
   Error loading index_faiss
   ```
   Solution: Delete `index_faiss` folder and rebuild

5. **Memory Issues**
   - Reduce chunk_size
   - Process fewer documents at once
   - Use smaller embedding model

## 📊 Performance Optimization

### Speed
- Cache LLM initialization (`@st.cache_resource`)
- Pre-build FAISS index
- Optimize chunk size
- Limit retrieval documents

### Accuracy
- Use quality source documents
- Tune chunk overlap
- Adjust retriever k value
- Refine system prompts

### Cost
- Use Claude Sonnet instead of Opus
- Implement response caching
- Optimize prompt length
- Monitor token usage

## 🔄 Updates & Maintenance

### Updating Documents
1. Add new PDFs to `documents` folder
2. Delete `index_faiss` folder
3. Restart application (auto-rebuilds index)

### Model Updates
- Check Anthropic's model releases
- Update model name in code
- Test thoroughly before production

## 📈 Roadmap

### Phase 1 ✅
- [x] Claude API integration
- [x] Basic RAG implementation
- [x] Console interface
- [x] Streamlit web interface

### Phase 2 (Planned)
- [ ] Multi-format support (DOC, DOCX, TXT, CSV)
- [ ] Advanced search (filters, facets)
- [ ] User authentication
- [ ] Admin dashboard

### Phase 3 (Future)
- [ ] Multi-language support
- [ ] Voice interface
- [ ] Mobile app
- [ ] Analytics & reporting

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- Additional document format support
- Enhanced UI/UX
- Performance optimizations
- Security features
- Testing coverage

## 📄 License

This project is provided as-is for healthcare IT support use cases.

## 🆘 Support

For issues or questions:
1. Check troubleshooting section
2. Review Anthropic documentation: https://docs.anthropic.com/
3. Check LangChain docs: https://python.langchain.com/

## 🙏 Acknowledgments

- **Anthropic** for Claude API
- **LangChain** for RAG framework
- **HuggingFace** for embeddings
- **Meta** for FAISS
- **Streamlit** for UI framework

## 📞 Contact

For healthcare-specific implementations or customizations, please consult with your IT security and compliance teams.

---

**Version**: 1.0.0 (Claude API)  
**Last Updated**: November 2025  
**Status**: Production Ready
