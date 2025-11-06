# Quick Start Guide - IT Support Chatbot with Claude API

Get up and running in 5 minutes!

## Prerequisites
- Python 3.9+ installed
- Anthropic API key ([Get one here](https://console.anthropic.com/))

## Setup (3 steps)

### 1. Install Dependencies
```bash
pip install langchain langchain-anthropic langchain-community langchain-huggingface faiss-cpu sentence-transformers PyMuPDF anthropic streamlit python-dotenv
```

### 2. Configure API Key
Create a file named `.env`:
```
ANTHROPIC_API_KEY=sk-ant-your-actual-key-here
```

### 3. Add Documents
```bash
mkdir documents
# Copy your PDF files to the documents folder
```

## Run

### Web Interface (Recommended)
```bash
streamlit run app_claude.py
```
Opens at: http://localhost:8501

### Console Interface
```bash
python it_support_chatbot_claude_api.py console
```

### Test Mode
```bash
python it_support_chatbot_claude_api.py test
```

## First Time Usage

1. **Start the application** using one of the methods above
2. **Wait for indexing** (first run only - indexes your PDFs)
3. **Ask a question** like:
   - "How do I reset my password?"
   - "What's the process for VPN access?"
   - "How to install software?"

## Docker (Alternative)

If you prefer Docker:

```bash
# 1. Build
docker-compose build

# 2. Configure .env file (same as above)

# 3. Add documents to ./documents folder

# 4. Run
docker-compose up -d

# 5. Access at http://localhost:8501
```

## Troubleshooting

**"API key not found"**
→ Check your `.env` file exists and has the correct key

**"No documents found"**
→ Add PDF files to the `documents` folder

**"Import error"**
→ Run: `pip install -r requirements.txt`

## What's Next?

- Read **README.md** for detailed information
- Check **SETUP_GUIDE.md** for step-by-step instructions
- Review **PROJECT_SUMMARY.md** for architecture details

## Support

For issues, check the full documentation in README.md or SETUP_GUIDE.md

---

**That's it!** You're ready to use your IT Support Chatbot. 🎉
