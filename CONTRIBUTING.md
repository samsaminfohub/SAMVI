# Contributing to IT Support Chatbot

Thank you for your interest in contributing to the IT Support Chatbot project!

## How to Contribute

### Reporting Issues

1. Check if the issue already exists
2. Use the issue template
3. Provide detailed information:
   - Python version
   - OS and version
   - Steps to reproduce
   - Expected vs actual behavior
   - Error messages/logs

### Submitting Changes

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes**
   - Follow the code style
   - Add tests if applicable
   - Update documentation
4. **Commit your changes**
   ```bash
   git commit -m "Add: brief description of changes"
   ```
5. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```
6. **Create a Pull Request**

## Code Style

### Python
- Follow PEP 8
- Use type hints
- Add docstrings to functions
- Maximum line length: 100 characters

### Example
```python
def load_llm(model_name: str, temperature: float = 0.7) -> ChatAnthropic:
    """
    Initialize Claude LLM with specified parameters.
    
    Args:
        model_name: Claude model identifier
        temperature: Controls randomness (0.0-1.0)
    
    Returns:
        Configured Claude LLM instance
    """
    # Implementation
```

## Testing

Run tests before submitting:
```bash
pytest tests/ -v
```

## Documentation

- Update README.md if adding features
- Add docstrings to new functions
- Update relevant documentation in docs/

## Commit Message Guidelines

Format: `Type: Brief description`

Types:
- `Add:` New feature
- `Fix:` Bug fix
- `Update:` Update existing functionality
- `Docs:` Documentation changes
- `Test:` Add or update tests
- `Refactor:` Code refactoring
- `Style:` Code style changes

Examples:
- `Add: Support for DOCX file format`
- `Fix: Memory leak in document indexing`
- `Update: Improve retriever accuracy`

## Pull Request Process

1. Update documentation
2. Add tests for new features
3. Ensure all tests pass
4. Update CHANGELOG.md
5. Request review from maintainers

## Code Review

Reviewers will check:
- Code quality and style
- Test coverage
- Documentation
- Security implications
- Performance impact

## Questions?

- Open an issue for discussion
- Check existing issues and PRs
- Review documentation

Thank you for contributing!
