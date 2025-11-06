# GitHub Repository Setup Guide

Complete guide for setting up your IT Support Chatbot repository on GitHub.

## 📋 Table of Contents
1. [Quick Setup](#quick-setup)
2. [Detailed Setup](#detailed-setup)
3. [Repository Configuration](#repository-configuration)
4. [GitHub Actions Setup](#github-actions-setup)
5. [Collaboration Setup](#collaboration-setup)
6. [Deployment Options](#deployment-options)

---

## Quick Setup

### Option 1: Using GitHub Web Interface

1. **Create new repository** on GitHub
   - Go to https://github.com/new
   - Name: `it-support-chatbot-claude`
   - Description: `Healthcare IT Support Chatbot using Claude API and RAG`
   - Choose Public or Private
   - Don't initialize with README (we have one)
   - Click "Create repository"

2. **Upload project files**
   - Download the `it-support-chatbot-claude` folder
   - Use GitHub's upload feature or:

3. **Initialize from command line**:
   ```bash
   cd it-support-chatbot-claude
   git init
   git add .
   git commit -m "Initial commit: IT Support Chatbot with Claude API"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/it-support-chatbot-claude.git
   git push -u origin main
   ```

### Option 2: Using GitHub CLI

```bash
cd it-support-chatbot-claude

# Initialize git
git init
git add .
git commit -m "Initial commit: IT Support Chatbot with Claude API"

# Create repo and push
gh repo create it-support-chatbot-claude --public --source=. --remote=origin --push
```

---

## Detailed Setup

### Step 1: Prepare Local Repository

```bash
# Navigate to project
cd /path/to/it-support-chatbot-claude

# Initialize git
git init

# Add all files
git add .

# Check what will be committed
git status

# Make initial commit
git commit -m "Initial commit: IT Support Chatbot with Claude API v1.0.0"
```

### Step 2: Create GitHub Repository

#### Via Web Interface:
1. Go to https://github.com/new
2. Fill in details:
   - **Repository name**: `it-support-chatbot-claude`
   - **Description**: `Healthcare IT Support Chatbot using Claude API and RAG`
   - **Visibility**: Choose Public or Private
   - **DO NOT** check "Initialize with README"
   - **DO NOT** add .gitignore (we have one)
   - **DO NOT** choose a license (we have MIT)
3. Click "Create repository"

#### Via GitHub CLI:
```bash
gh repo create it-support-chatbot-claude \
  --public \
  --description "Healthcare IT Support Chatbot using Claude API and RAG" \
  --source=. \
  --remote=origin
```

### Step 3: Connect and Push

```bash
# Add remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/it-support-chatbot-claude.git

# Verify remote
git remote -v

# Push to GitHub
git branch -M main
git push -u origin main
```

### Step 4: Verify Upload

Check on GitHub:
- ✅ All files uploaded
- ✅ README.md displays properly
- ✅ .gitignore working (no .env, venv/, etc.)
- ✅ Folder structure correct

---

## Repository Configuration

### Step 1: Set Repository Description

On GitHub repository page:
1. Click ⚙️ Settings
2. Update description: `Healthcare IT Support Chatbot using Claude API and RAG`
3. Add website URL (if deploying)
4. Add topics/tags:
   - `chatbot`
   - `claude-api`
   - `rag`
   - `healthcare`
   - `it-support`
   - `langchain`
   - `anthropic`
   - `streamlit`

### Step 2: Configure Branch Protection

Settings → Branches → Add rule:
- Branch name pattern: `main`
- ✅ Require pull request reviews before merging
- ✅ Require status checks to pass before merging
- ✅ Require branches to be up to date before merging
- ✅ Include administrators

### Step 3: Set Up GitHub Pages (Optional)

For documentation hosting:
1. Settings → Pages
2. Source: Deploy from branch
3. Branch: `main`
4. Folder: `/docs`
5. Save

Access docs at: `https://YOUR_USERNAME.github.io/it-support-chatbot-claude/`

---

## GitHub Actions Setup

### Step 1: Enable GitHub Actions

1. Go to repository Settings
2. Actions → General
3. Enable "Allow all actions and reusable workflows"
4. Click Save

### Step 2: Add Repository Secrets

For CI/CD to work with Docker Hub:

Settings → Secrets and variables → Actions → New repository secret:

1. **DOCKER_USERNAME**
   - Your Docker Hub username
   
2. **DOCKER_PASSWORD**
   - Your Docker Hub password or access token

3. **ANTHROPIC_API_KEY** (optional, for testing)
   - Your Anthropic API key

### Step 3: Verify Workflow

After pushing code:
1. Go to Actions tab
2. Check CI/CD Pipeline workflow
3. First run should start automatically
4. Verify all jobs pass:
   - ✅ Test (Python 3.9, 3.10, 3.11)
   - ✅ Docker build
   - ✅ Security scan

---

## Collaboration Setup

### Step 1: Add Collaborators

Settings → Collaborators → Add people

Role options:
- **Read**: View only
- **Triage**: Manage issues and PRs
- **Write**: Push to repository
- **Maintain**: Manage repository
- **Admin**: Full access

### Step 2: Set Up Issues

Create issue templates:

`.github/ISSUE_TEMPLATE/bug_report.md`:
```markdown
---
name: Bug Report
about: Create a report to help us improve
title: '[BUG] '
labels: bug
assignees: ''
---

**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. See error

**Expected behavior**
What you expected to happen.

**Environment:**
 - OS: [e.g. Windows 10]
 - Python Version: [e.g. 3.11]
 - Browser: [e.g. Chrome]

**Additional context**
Any other context about the problem.
```

### Step 3: Set Up Pull Request Template

`.github/PULL_REQUEST_TEMPLATE.md`:
```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests added/updated
- [ ] Documentation updated
- [ ] All tests pass
- [ ] CHANGELOG.md updated
```

---

## Deployment Options

### Option 1: GitHub Pages (Documentation)

Already covered in Repository Configuration.

### Option 2: Docker Hub Integration

Automatic via GitHub Actions (already configured).

Images pushed to:
```
YOUR_DOCKER_USERNAME/it-support-chatbot:latest
YOUR_DOCKER_USERNAME/it-support-chatbot:COMMIT_SHA
```

### Option 3: Cloud Platforms

#### Heroku
```bash
heroku create it-support-chatbot
git push heroku main
```

#### AWS
Use AWS Elastic Beanstalk or ECS with the Docker image.

#### Azure
Use Azure Container Instances or App Service.

#### Google Cloud
Use Google Cloud Run or GKE.

---

## Repository Maintenance

### Regular Tasks

**Weekly:**
```bash
# Update dependencies
pip list --outdated
pip install --upgrade [package]

# Update requirements.txt
pip freeze > requirements.txt

# Commit changes
git add requirements.txt
git commit -m "Update: Dependencies upgraded"
git push
```

**Monthly:**
- Review and close stale issues
- Update documentation
- Check security alerts
- Review pull requests

**Quarterly:**
- Major version updates
- Performance optimization
- Security audit
- Feature planning

### Version Tagging

When releasing new versions:

```bash
# Tag version
git tag -a v1.0.0 -m "Release version 1.0.0"

# Push tag
git push origin v1.0.0

# Create release on GitHub
gh release create v1.0.0 \
  --title "Version 1.0.0" \
  --notes "Initial release with Claude API integration"
```

---

## Monitoring

### GitHub Insights

Check regularly:
1. **Traffic**: Views and clones
2. **Contributions**: Commits and contributors
3. **Dependencies**: Security alerts
4. **Actions**: Workflow runs

### Security

Enable:
1. **Dependabot alerts**: Auto-detect vulnerable dependencies
2. **Code scanning**: Find security vulnerabilities
3. **Secret scanning**: Prevent exposed secrets

Settings → Security:
- ✅ Dependency graph
- ✅ Dependabot alerts
- ✅ Dependabot security updates
- ✅ Code scanning
- ✅ Secret scanning

---

## Troubleshooting

### Common Issues

**Push rejected:**
```bash
# Pull first
git pull origin main --rebase

# Then push
git push origin main
```

**Large files:**
```bash
# Remove from history
git filter-branch --force --index-filter \
  "git rm --cached --ignore-unmatch PATH/TO/FILE" \
  --prune-empty --tag-name-filter cat -- --all
```

**Authentication:**
```bash
# Use personal access token
git remote set-url origin https://TOKEN@github.com/USERNAME/REPO.git
```

---

## Best Practices

### Commit Messages

Format:
```
Type: Brief description

Detailed explanation (optional)

Closes #issue_number (if applicable)
```

Types:
- `Add:` New feature
- `Fix:` Bug fix
- `Update:` Update existing feature
- `Docs:` Documentation
- `Test:` Tests
- `Refactor:` Code refactoring
- `Style:` Formatting

### Branching Strategy

```
main            # Production-ready code
  └── develop   # Development branch
      └── feature/your-feature  # Feature branches
```

Workflow:
1. Create feature branch from `develop`
2. Work on feature
3. Create PR to `develop`
4. After review, merge to `develop`
5. When ready for release, merge `develop` to `main`

---

## Additional Resources

- [GitHub Documentation](https://docs.github.com)
- [GitHub Actions](https://docs.github.com/en/actions)
- [Git Best Practices](https://git-scm.com/book/en/v2)
- [Semantic Versioning](https://semver.org/)

---

## Summary Checklist

- [ ] Repository created on GitHub
- [ ] All files pushed
- [ ] README displays correctly
- [ ] .gitignore working
- [ ] Branch protection enabled
- [ ] GitHub Actions working
- [ ] Secrets configured
- [ ] Collaborators added (if any)
- [ ] Issue templates created
- [ ] PR template created
- [ ] First release tagged
- [ ] Security features enabled
- [ ] Documentation reviewed

---

**Your repository is now ready for collaboration and deployment!** 🎉

For questions or issues, refer to the main README.md or create an issue on GitHub.

---

**Last Updated**: November 2025  
**Version**: 1.0.0
