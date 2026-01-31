# Blog Post Automation Tool

Automates blog post creation from CSV topics with AI-powered content generation using local LLM (Ollama/Llama). 

## Workflow

```
┌─────────────────────────────┐
│ Read CSV file               │
│ (title, excerpt, tags, date)│
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Filter by today's date      │
│ (only process matching rows)│
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Call Local LLM (Ollama)     │
│ Generate blog post content  │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Format into blog structure  │
│ (JSON with HTML content)    │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Create feature branch       │
│ Add post to posts.js        │
└──────────────┬──────────────┘
               │
               ▼
┌─────────────────────────────┐
│ Commit & Push to remote     │
│ Create PR to main branch    │
└─────────────────────────────┘
```

## Features

✅ Reads blog topics from **CSV file** with headers  
✅ Filters rows by **today's date** only  
✅ Generates blog content using **local LLM** (Ollama/Llama)  
✅ Formats into required blog post structure  
✅ Commits to feature branches automatically  
✅ Creates pull requests against main branch  

## Project Structure

```
blog_automation/
├── blog_automation.py       # Main automation script
├── create_sample_csv.py     # Helper to generate sample CSV file
├── requirements.txt         # Python dependencies
├── .env.example            # Configuration template
└── README.md               # This file
```

## Files Overview

| File | Purpose |
|------|---------|
| `blog_automation.py` | Core automation engine - reads CSV, calls LLM, creates PRs |
| `create_sample_csv.py` | Generates sample CSV with test topics (optional) |
| `requirements.txt` | Dependencies: GitPython, python-dotenv, requests |
| `.env.example` | Template for GitHub & LLM configuration |
| `topics.csv` | Your actual blog topics (user-created) |

**Note:** Only `.py`, `requirements.txt`, and `.env` files are needed for production use.

## Setup

### 1. Install Dependencies

```bash
cd blog_automation
pip install -r requirements.txt
```

### 2. Setup Local LLM (Ollama)

Download and run [Ollama](https://ollama.ai):

```bash
# Install Ollama from https://ollama.ai
# Run Ollama server
ollama serve

# In another terminal, pull a model
ollama pull llama2
# or
ollama pull mistral
```

The LLM API will be available at `http://localhost:11434/api/generate`

### 3. Configure Environment

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env`:

```env
# GitHub Configuration (required)
GITHUB_TOKEN=ghp_xxxxxxxxxxxx
GITHUB_OWNER=your_github_username
GITHUB_REPO=your_repo_name

# Blog Configuration
BLOG_AUTHOR=Saravana Kumar
BASE_BRANCH=main
FEATURE_BRANCH_PREFIX=feature/blog-

# Local LLM Configuration
LLM_ENDPOINT=http://localhost:11434/api/generate
LLM_MODEL=llama2  # or mistral, neural-chat, etc.
```

### 4. Prepare CSV File

Create a CSV file with headers and data. **The script only processes rows where the date column matches today's date.**

**Option A: Create CSV Manually**

Create `topics.csv`:
```csv
title,excerpt,tags,author,date
Getting Started with Docker,Learn Docker containerization basics,docker,devops,containers,Saravana Kumar,2025-01-31
Advanced Python Patterns,Master design patterns in Python,python,patterns,programming,Saravana Kumar,2025-01-31
Database Optimization,Improve database performance,database,sql,optimization,Saravana Kumar,2025-02-01
```

**Option B: Generate Sample CSV (for testing - optional)**

```bash
python create_sample_csv.py
# Creates: sample_topics.csv with 4 sample topics
# Topics with today's date will be processed
# Topics with future dates will be skipped
```

**Required CSV Columns:**

| Column | Required | Format | Example |
|--------|----------|--------|---------|
| `title` | Yes | String | "Getting Started with Docker" |
| `excerpt` | Yes | String | "Learn Docker basics..." |
| `tags` | Yes | Comma-separated | "docker,devops,containers" |
| `date` | Yes | YYYY-MM-DD | "2025-01-31" |
| `author` | No | String | "Saravana Kumar" |

**Important:** Only rows with `date` = today's date will be processed!

## Usage

```bash
python blog_automation.py <path_to_csv_file>
```

### Example

```bash
# Run with sample CSV
python blog_automation.py topics.csv
```

### Sample Output

```
Reading topics from CSV file...
CSV Headers found: ['title', 'excerpt', 'tags', 'author', 'date']
  Row 1: Processing 'Getting Started with Docker' (date: 2025-01-31)
  Row 2: Skipping (date 2025-02-01 != today 2025-01-31)

============================================================
Processing topic 1/1: Getting Started with Docker
============================================================
  Generating content via LLM...
  Calling LLM at http://localhost:11434/api/generate...
  LLM generated 1245 characters of content
✓ Created and checked out branch: feature/blog-getting-started--1706745600
✓ Added post to posts.js
✓ Committed changes: feat: Add blog post - Getting Started with Docker
✓ Pushed branch to remote: feature/blog-getting-started--1706745600
✓ Created pull request: https://github.com/user/repo/pull/42
✓ Successfully created PR: https://github.com/user/repo/pull/42
```

## Workflow

For each topic in CSV with today's date:

1. ✅ Reads title, excerpt, tags, author, date from CSV headers
2. ✅ Calls local LLM to generate detailed blog post HTML content
3. ✅ Creates feature branch: `feature/blog-{title}-{timestamp}`
4. ✅ Adds post to `posts.js` in required JSON format
5. ✅ Commits with message: `feat: Add blog post - {title}`
6. ✅ Pushes branch to GitHub remote
7. ✅ Creates PR against main branch

## LLM Models Supported

Any model compatible with Ollama:

- `llama2` - Meta's Llama 2 (default)
- `mistral` - Mistral 7B (faster, good quality)
- `neural-chat` - Intel Neural Chat
- `openchat` - OpenChat
- `dolphin-mixtral` - Dolphin Mixtral

Pull models:
```bash
ollama pull mistral
ollama pull neural-chat
```

## GitHub Configuration

### Generate GitHub Token

1. Go to [GitHub Settings → Personal Access Tokens](https://github.com/settings/tokens)
2. Click "Generate new token" → "Generate new token (classic)"
3. Select scopes: `repo` (full control of private repositories)
4. Generate and copy the token
5. Add to `.env`: `GITHUB_TOKEN=ghp_xxxxxxxxxxxx`

## CSV Format Details

### Supported Columns

| Column | Required | Format | Example |
|--------|----------|--------|---------|
| `title` | Yes | String | "Getting Started with Docker" |
| `excerpt` | Yes | String | "Learn Docker basics..." |
| `tags` | Yes | Comma-separated | "docker,devops,containers" |
| `author` | No | String | "Saravana Kumar" |
| `date` | Yes | YYYY-MM-DD | "2025-01-31" |

### Date Filtering Logic

- Only rows where `date` = today's date (YYYY-MM-DD) are processed
- Other rows are skipped with a log message
- This allows scheduling topics in advance

## Troubleshooting

### "Cannot connect to LLM at http://localhost:11434"

```bash
# Make sure Ollama is running
ollama serve
```

### "LLM request timed out"

- Increase timeout or reduce model complexity
- Try a smaller, faster model like `mistral`

### Git errors

```bash
# Ensure git is configured
git config --global user.name "Your Name"
git config --global user.email "your@email.com"

# Ensure origin remote exists
git remote -v
```

### CSV not found

```bash
# Use absolute path or ensure file is in repo directory
python blog_automation.py /full/path/to/topics.csv
```

## Generated Post Format

Posts are added to `posts.js` in this structure:

```javascript
{
  "id": "post-1706745600",
  "title": "Blog Post Title",
  "author": "Author Name",
  "date": "2025-01-31",
  "tags": ["tag1", "tag2"],
  "image": "https://images.unsplash.com/...",
  "placeholder": "https://images.unsplash.com/...",
  "excerpt": "Short summary...",
  "content": "<p>LLM-generated HTML content...</p>"
}
```

## Advanced Configuration

### Use Different LLM Models

Change in `.env`:
```env
LLM_MODEL=mistral
```

### Adjust LLM Parameters

Edit `blog_automation.py`, find `call_local_llm()`:
```python
payload = {
    "model": self.llm_model,
    "temperature": 0.7,  # Lower = more focused, Higher = more creative
    "top_p": 0.9,        # Nucleus sampling
    "top_k": 40,         # Top-k sampling
}
```

## Notes

- Generated posts can be reviewed and edited in the PR before merging
- Each topic creates a separate PR for independent review
- Timestamps are included in branch names to avoid conflicts
- CSV headers must match column names exactly (case-sensitive)
- Only rows with today's date are processed


## How to Get GitHub Details

### 1. GITHUB_TOKEN (Personal Access Token)

**Steps:**
1. Go to GitHub → Settings (top right profile menu)
2. Click "Developer settings" (bottom left sidebar)
3. Click "Personal access tokens" → "Tokens (classic)"
4. Click "Generate new token" → "Generate new token (classic)"
5. Name: "Blog Automation"
6. Select scopes:
   - ✅ repo (full control of private repositories)
   - ✅ workflow (update GitHub Action workflows)
7. Click "Generate token"
8. Copy the token (you won't see it again!)

### 2. GITHUB_OWNER (Your Username)

Simply your GitHub username!

### 3. GITHUB_REPO (Repository Name)

Your repository name (without .git)

### Complete Configuration

```env
# GitHub Configuration
GITHUB_TOKEN=ghp_your_token_here
GITHUB_OWNER=Itzsaravana
GITHUB_REPO=Itzsaravana.github.io

# Blog Configuration
BLOG_AUTHOR=Saravana Kumar
BASE_BRANCH=main
FEATURE_BRANCH_PREFIX=feature/blog-

# Local LLM Configuration
LLM_ENDPOINT=http://localhost:11434/api/generate
LLM_MODEL=llama2
```