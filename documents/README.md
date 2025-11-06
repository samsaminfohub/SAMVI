# Documents Folder

This folder contains PDF documents that serve as the knowledge base for the IT Support Chatbot.

## Usage

1. **Add your PDF files** here
2. The application will automatically:
   - Extract text from PDFs
   - Split into chunks
   - Create embeddings
   - Index in vector store

## Document Guidelines

### What to Include
- IT support manuals
- Troubleshooting guides
- FAQ documents
- System access procedures
- Security policies
- Software installation guides
- Common issue resolutions

### Best Practices

#### Format
- ✅ PDF format (text-based, not scanned images)
- ✅ Clear section headings
- ✅ Well-structured content
- ✅ Tables of contents

#### Content Quality
- ✅ Accurate and up-to-date information
- ✅ Step-by-step instructions
- ✅ Clear problem descriptions
- ✅ Contact information for escalation

#### Organization
- ✅ Logical file names (e.g., `password-reset-guide.pdf`)
- ✅ Grouped by topic if many documents
- ✅ Version control in filename (e.g., `manual-v2.1.pdf`)

### Example Structure

```
documents/
├── password-reset-guide.pdf
├── vpn-access-instructions.pdf
├── software-installation-procedures.pdf
├── troubleshooting-common-issues.pdf
├── security-policies.pdf
└── escalation-procedures.pdf
```

## Healthcare Considerations

### ⚠️ Important
- **NO PHI** (Protected Health Information)
- **NO PII** (Personally Identifiable Information)
- **NO sensitive passwords** or credentials
- **NO patient data**

### What's Safe to Include
- General IT procedures
- Generic troubleshooting steps
- Software installation guides
- System access procedures (without credentials)
- Security policies (public-facing)

## Testing Your Documents

After adding documents:

1. Run the application
2. Wait for indexing to complete
3. Ask test questions like:
   - "How do I reset my password?"
   - "What's the VPN setup process?"
   - "How to install software X?"

## Updating Documents

When you update or add documents:

1. **Delete the index**: `rm -rf index_faiss/`
2. **Restart the application**: It will rebuild the index automatically

Or use force rebuild:
```python
retriever = config_retriever(force_rebuild=True)
```

## Document Statistics

After indexing, the application will show:
- Number of PDFs loaded
- Number of chunks created
- Index size

## Troubleshooting

### "No documents found"
- Check that PDF files are in this folder
- Verify file permissions
- Ensure files have `.pdf` extension

### "Failed to extract text"
- File might be image-based (scanned)
- File might be password-protected
- File might be corrupted

### Poor answer quality
- Documents might be too general
- Add more specific documentation
- Improve document structure
- Increase chunk overlap

## Need Help?

Check the main documentation:
- README.md
- SETUP_GUIDE.md
- DOCKER_DEPLOYMENT.md

---

**Note**: This folder is included in `.gitignore` to prevent accidentally committing sensitive documents to version control.
