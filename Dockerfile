# Multi-stage Dockerfile for IT Support Chatbot with Claude API

# Stage 1: Base image with Python and dependencies
FROM python:3.11-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*
	

# Stage 2: Dependencies installation
FROM base as dependencies

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
	
RUN pip uninstall langchain langchain-core langchain-community langchain-anthropic langchain-huggingface langchain-text-splitters -y
RUN pip install langchain
RUN pip install langchain-anthropic langchain-community langchain-huggingface langchain-text-splitters
RUN pip install langchain langchain-anthropic langchain-community langchain-huggingface langchain-text-splitters
RUN pip install streamlit faiss-cpu pymupdf python-dotenv sentence-transformers
RUN pip install tf-keras


# Stage 3: Application
FROM dependencies as application

# Copy application files
COPY src/it_support_chatbot_claude_api.py ./
COPY src/app_claude.py ./

# Create necessary directories
RUN mkdir -p documents index_faiss

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Create entrypoint script
RUN echo '#!/bin/bash\n\
if [ "$1" = "web" ]; then\n\
    echo "Starting Streamlit web interface..."\n\
    streamlit run app_claude.py\n\
elif [ "$1" = "console" ]; then\n\
    echo "Starting console interface..."\n\
    python it_support_chatbot_claude_api.py console\n\
elif [ "$1" = "test" ]; then\n\
    echo "Running tests..."\n\
    python it_support_chatbot_claude_api.py test\n\
else\n\
    echo "Usage: docker run [image] [web|console|test]"\n\
    echo "Default: web"\n\
    streamlit run app_claude.py\n\
fi' > /app/entrypoint.sh && \
    chmod +x /app/entrypoint.sh

# Use entrypoint
ENTRYPOINT ["/app/entrypoint.sh"]

# Default command
CMD ["web"]

# Labels
LABEL maintainer="IT Support Team"
LABEL description="IT Support Chatbot for Healthcare using Claude API"
LABEL version="1.0.0"
