# IT Support Chatbot - Complete Setup Guide

This guide provides detailed step-by-step instructions for setting up the IT Support Chatbot using Claude API.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation Steps](#installation-steps)
3. [Configuration](#configuration)
4. [Document Preparation](#document-preparation)
5. [Running the Application](#running-the-application)
6. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements
- **Operating System**: Windows 10/11, macOS 10.15+, or Linux (Ubuntu 20.04+)
- **Python**: Version 3.9 or higher
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: At least 5GB free space

### Required Accounts
- **Anthropic Account**: For Claude API access (https://console.anthropic.com/)

---

## Installation Steps

### Step 1: Install Python

Verify Python installation:
```bash
python --version
# Should show Python 3.9 or higher
```

### Step 2: Create Project Directory

```bash
mkdir it-support-chatbot
cd it-support-chatbot
```

### Step 3: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Configuration

### Get Your Anthropic API Key

1. Go to https://console.anthropic.com/
2. Create an API key
3. Copy the key (starts with `sk-ant-`)

### Create .env File

Create `.env` in project root:

```
ANTHROPIC_API_KEY=your-key-here
```

---

## Document Preparation

### Create Documents Folder

```bash
mkdir documents
```

### Add Your PDF Documents

Place IT support PDFs in the `documents` folder:
- Troubleshooting guides
- FAQ documents
- System manuals

---

## Running the Application

### Console Interface

```bash
python it_support_chatbot_claude_api.py console
```

### Web Interface (Streamlit)

```bash
streamlit run app_claude.py
```

Opens at `http://localhost:8501`

---

## Troubleshooting

### API Key Not Found
- Check `.env` file exists
- Verify API key is correct
- Ensure no extra spaces

### No Documents Found
- Check PDFs are in `documents` folder
- Verify folder path is correct

### Installation Errors
```bash
# Upgrade pip
pip install --upgrade pip

# Reinstall packages
pip install -r requirements.txt --force-reinstall
```

---

For more details, see README.md
