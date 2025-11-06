# Project Structure

Complete directory structure for the IT Support Chatbot with Claude API.

## Directory Tree

```
it-support-chatbot-claude/
│
├── .github/                          # GitHub specific files
│   └── workflows/
│       └── ci-cd.yml                # GitHub Actions CI/CD pipeline
│
├── config/                           # Configuration files
│   └── .env.template                # Environment variables template
│
├── docs/                             # Documentation
│   ├── DOCKER_DEPLOYMENT.md         # Docker deployment guide
│   ├── FILE_INDEX.md                # Complete file listing
│   ├── PROJECT_SUMMARY.md           # Project overview
│   ├── QUICKSTART.md                # Quick start guide
│   └── SETUP_GUIDE.md               # Detailed setup instructions
│
├── documents/                        # PDF knowledge base
│   └── README.md                    # Documents folder guide
│
├── scripts/                          # Utility scripts
│   ├── README.md                    # Scripts documentation
│   └── setup.sh                     # Unix/Linux/macOS setup script
│
├── src/                              # Source code
│   ├── app_claude.py                # Streamlit web interface
│   └── it_support_chatbot_claude_api.py  # Main application
│
├── tests/                            # Test files
│   ├── README.md                    # Testing guide
│   ├── conftest.py                  # Pytest configuration
│   └── test_basic.py                # Basic unit tests
│
├── .gitignore                        # Git ignore rules
├── CHANGELOG.md                      # Version history
├── CONTRIBUTING.md                   # Contribution guidelines
├── Dockerfile                        # Docker image definition
├── LICENSE                           # MIT License
├── README.md                         # Main documentation
├── docker-compose.yml                # Docker Compose configuration
└── requirements.txt                  # Python dependencies
```

## Directory Descriptions

### Root Level

#### `.github/`
GitHub-specific configuration files.
- **workflows/ci-cd.yml**: Automated testing, building, and deployment pipeline

#### `config/`
Configuration templates and settings.
- **.env.template**: Template for environment variables (API keys, settings)

#### `docs/`
Comprehensive project documentation.
- All user guides and technical documentation
- Keep markdown format for GitHub compatibility

#### `documents/`
PDF knowledge base for the chatbot.
- Add your IT support PDFs here
- Automatically indexed by the application
- Protected by .gitignore (won't commit to repo)

#### `scripts/`
Automation and utility scripts.
- Setup scripts
- Maintenance utilities
- Deployment helpers

#### `src/`
Application source code.
- **app_claude.py**: Streamlit web interface
- **it_support_chatbot_claude_api.py**: Core chatbot logic

#### `tests/`
Unit and integration tests.
- Pytest-based testing
- CI/CD integration
- Coverage reporting

### Configuration Files

#### `.gitignore`
Specifies files/folders to exclude from version control:
- Environment files (.env)
- Python cache (__pycache__)
- Virtual environments (venv/)
- Vector indexes (index_faiss/)
- Sensitive documents

#### `CHANGELOG.md`
Version history and release notes.
- Follows Keep a Changelog format
- Documents all changes by version

#### `CONTRIBUTING.md`
Guidelines for contributors.
- How to report issues
- Code style guidelines
- Pull request process

#### `Dockerfile`
Container image definition.
- Multi-stage build
- Python 3.11 base
- Production-ready

#### `LICENSE`
MIT License for the project.

#### `README.md`
Main project documentation.
- Features overview
- Installation guide
- Usage instructions
- Configuration details

#### `docker-compose.yml`
Multi-container orchestration.
- Service definitions
- Volume mounts
- Network configuration

#### `requirements.txt`
Python package dependencies.
- All required packages
- Version specifications

## File Purposes

### Core Application

| File | Purpose | Lines | Type |
|------|---------|-------|------|
| src/it_support_chatbot_claude_api.py | Main chatbot logic | ~480 | Python |
| src/app_claude.py | Web interface | ~330 | Python |

### Configuration

| File | Purpose | Size | Type |
|------|---------|------|------|
| requirements.txt | Dependencies | ~20 | Text |
| config/.env.template | Config template | Small | Text |
| docker-compose.yml | Docker config | ~60 | YAML |
| Dockerfile | Container def | ~70 | Docker |

### Documentation

| File | Purpose | Size | Type |
|------|---------|------|------|
| README.md | Main docs | ~500 | Markdown |
| docs/QUICKSTART.md | Quick guide | ~100 | Markdown |
| docs/SETUP_GUIDE.md | Setup guide | ~300 | Markdown |
| docs/DOCKER_DEPLOYMENT.md | Docker guide | ~400 | Markdown |
| docs/PROJECT_SUMMARY.md | Overview | ~600 | Markdown |
| docs/FILE_INDEX.md | File listing | ~300 | Markdown |

### Development

| File | Purpose | Lines | Type |
|------|---------|-------|------|
| .github/workflows/ci-cd.yml | CI/CD | ~90 | YAML |
| tests/conftest.py | Test config | ~40 | Python |
| tests/test_basic.py | Unit tests | ~120 | Python |
| scripts/setup.sh | Setup script | ~100 | Bash |

## Generated Directories (Not in Repo)

These are created during runtime:

```
it-support-chatbot-claude/
├── venv/                    # Virtual environment (created by setup)
├── index_faiss/             # Vector store (auto-generated)
│   ├── index.faiss
│   └── index.pkl
├── logs/                    # Application logs (optional)
└── .env                     # Environment variables (user creates)
```

## Workflow

### Development Workflow

1. **Clone/Download** → Get project files
2. **Setup** → Run `scripts/setup.sh`
3. **Configure** → Edit `.env` with API key
4. **Develop** → Edit files in `src/`
5. **Test** → Run `pytest tests/`
6. **Commit** → Git commit and push
7. **CI/CD** → Automatic testing and building

### Deployment Workflow

1. **Local** → `streamlit run src/app_claude.py`
2. **Docker** → `docker-compose up -d`
3. **Cloud** → Push to cloud platform

## Adding New Features

### New Source File
Place in `src/` directory:
```
src/
└── new_feature.py
```

### New Tests
Place in `tests/` directory:
```
tests/
└── test_new_feature.py
```

### New Documentation
Place in `docs/` directory:
```
docs/
└── NEW_FEATURE_GUIDE.md
```

### New Scripts
Place in `scripts/` directory:
```
scripts/
└── new_automation.sh
```

## Best Practices

### File Organization
- ✅ Keep source code in `src/`
- ✅ Keep tests in `tests/`
- ✅ Keep docs in `docs/`
- ✅ Keep scripts in `scripts/`
- ✅ Keep configs in `config/`

### Naming Conventions
- ✅ Python files: `snake_case.py`
- ✅ Markdown files: `UPPERCASE.md` or `Title_Case.md`
- ✅ Scripts: `lowercase.sh`
- ✅ Configs: `.dotfile` or `lowercase.yml`

### Version Control
- ✅ Commit frequently
- ✅ Write clear commit messages
- ✅ Don't commit sensitive data
- ✅ Use .gitignore properly

## Maintenance

### Regular Updates
- Update dependencies in `requirements.txt`
- Update documentation when features change
- Keep CHANGELOG.md current
- Review and update .gitignore

### Cleanup
Remove generated files:
```bash
rm -rf venv/
rm -rf index_faiss/
rm -rf __pycache__/
rm -rf .pytest_cache/
```

## Size Information

| Category | Files | Size |
|----------|-------|------|
| Source Code | 2 | ~26 KB |
| Documentation | 6 | ~45 KB |
| Configuration | 5 | ~5 KB |
| Tests | 3 | ~5 KB |
| Scripts | 2 | ~3 KB |
| **Total** | **18** | **~84 KB** |

**Note**: Excludes dependencies, virtual environment, and generated files.

---

**Last Updated**: November 2025  
**Version**: 1.0.0
