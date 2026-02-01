#!/usr/bin/env python3
"""
Blog Post Automation Tool
Reads topics from Excel, generates blog posts, commits to feature branch, and creates PRs.
"""

import os
import sys
import json
import csv
import subprocess
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv
import requests
from git import Repo
from typing import List, Dict, Optional

# Load environment variables
load_dotenv()

class BlogAutomation:
    def __init__(self):
        self.github_token = os.getenv('GITHUB_TOKEN')
        self.github_owner = os.getenv('GITHUB_OWNER')
        self.github_repo = os.getenv('GITHUB_REPO')
        self.blog_author = os.getenv('BLOG_AUTHOR', 'Unknown')
        self.base_branch = os.getenv('BASE_BRANCH', 'main')
        self.feature_branch_prefix = os.getenv('FEATURE_BRANCH_PREFIX', 'feature/blog-')
        self.llm_endpoint = os.getenv('LLM_ENDPOINT', 'http://localhost:11434/api/generate')
        self.llm_model = os.getenv('LLM_MODEL', 'mistral')
        self.repo_path = Path(__file__).parent.parent
        self.repo = Repo(self.repo_path)
        
        # Validate required configs
        if not self.github_token or not self.github_owner or not self.github_repo:
            raise ValueError("Missing GitHub configuration. Please set GITHUB_TOKEN, GITHUB_OWNER, and GITHUB_REPO in .env")
    
    
    def read_topics_from_csv(self, csv_file: str) -> List[Dict]:
        """
        Read topics from CSV file with headers.
        Filters rows where date column matches current date.
        """
        topics = []
        current_date = datetime.now().strftime('%Y-%m-%d')
        
        try:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                
                if not reader.fieldnames:
                    raise ValueError("CSV file is empty or has no headers")
                
                print(f"CSV Headers found: {reader.fieldnames}")
                
                for row_idx, row in enumerate(reader, 1):
                    # Skip empty rows
                    if not any(row.values()):
                        continue
                    
                    # Check if date column exists and matches current date
                    date_value = row.get('date', '').strip() if 'date' in row else None
                    
                    if date_value and date_value != current_date:
                        print(f"  Row {row_idx}: Skipping (date {date_value} != today {current_date})")
                        continue
                    
                    topic = {
                        'title': row.get('title', '').strip(),
                        'excerpt': row.get('excerpt', '').strip(),
                        'tags': row.get('tags', '').strip(),
                        'author': row.get('author', self.blog_author).strip(),
                        'date': date_value or current_date,
                        'raw_data': row  # Store raw data for LLM processing
                    }
                    
                    if topic['title']:  # Only add if title is not empty
                        topics.append(topic)
                        print(f"  Row {row_idx}: Processing '{topic['title']}' (date: {topic['date']})")
            
            return topics
        except FileNotFoundError:
            print(f"Error: CSV file not found: {csv_file}")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            sys.exit(1)
    
    def call_local_llm(self, prompt: str) -> str:
        """
        Call local LLM (Ollama/Llama) via HTTP to generate content.
        """
        try:
            print(f"  Calling LLM at {self.llm_endpoint}...")
            print(f"  Using model: {self.llm_model}")
            print(f"  ⏳ This may take 1-5 minutes on first run...")
            
            payload = {
                "model": self.llm_model,
                "prompt": prompt,
                "stream": False,
                "temperature": 0.7,
            }
            
            # Increased timeout to 600 seconds (10 minutes)
            response = requests.post(self.llm_endpoint, json=payload, timeout=600)
            
            if response.status_code != 200:
                print(f"  ✗ Warning: LLM returned status {response.status_code}")
                print(f"    Response: {response.text[:200]}")
                return ""
            
            result = response.json()
            content = result.get('response', '').strip()
            
            if content:
                print(f"  ✓ LLM generated {len(content)} characters of content")
            else:
                print(f"  ⚠ LLM returned empty response")
            
            return content
        except requests.exceptions.ConnectionError:
            print(f"  ✗ Error: Cannot connect to LLM at {self.llm_endpoint}")
            print(f"  ✓ Solutions:")
            print(f"    1. Make sure Ollama is running: ollama serve")
            print(f"    2. Check endpoint in .env: LLM_ENDPOINT={self.llm_endpoint}")
            print(f"    3. Run: python test_ollama.py")
            return ""
        except requests.exceptions.Timeout:
            print(f"  ✗ Error: LLM request timed out after 10 minutes")
            print(f"  ✓ Solutions:")
            print(f"    1. Check if Ollama is responsive: curl http://localhost:11434/api/tags")
            print(f"    2. Try a faster model: ollama pull neural-chat")
            print(f"    3. Update .env: LLM_MODEL=neural-chat")
            print(f"    4. Increase system memory/resources")
            return ""
        except Exception as e:
            print(f"  ✗ Error calling LLM: {e}")
            return ""
    
    def get_fallback_image_url(self, tags: str) -> tuple:
        """
        Fetch image URLs from Unsplash based on tags/keywords.
        Returns tuple of (full_image_url, placeholder_url)
        """
        try:
            # Extract first tag as search keyword
            search_term = tags.split(',')[0].strip() if tags else 'technology'
            search_term = search_term.lower().replace(' ', '-')
            
            # Fallback for generic terms
            generic_terms = ['technology', 'blog', 'article', 'content', 'general']
            if search_term in generic_terms or len(search_term) < 2:
                search_term = 'technology'
            
            print(f"  Fetching image from Unsplash for tag: {search_term}")
            
            # Get API key from environment variable
            unsplash_api_key = os.getenv('UNSPLASH_API_KEY')
            
            if not unsplash_api_key:
                print(f"  ⚠ UNSPLASH_API_KEY not set in .env, using default images")
                return self._get_default_image_urls(search_term)
            
            # Unsplash API endpoint for random image
            unsplash_url = f"https://api.unsplash.com/photos/random?query={search_term}&orientation=landscape"
            headers = {
                'Accept-Version': 'v1',
                'Authorization': f'Client-ID {unsplash_api_key}'
            }
            
            response = requests.get(unsplash_url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                full_image_url = data['urls']['regular']
                placeholder_url = data['urls']['thumb']
                print(f"  ✓ Got image from Unsplash: {data['user']['name']}")
                return (full_image_url, placeholder_url)
            elif response.status_code == 401:
                print(f"  ⚠ Unsplash API error: 401 Unauthorized")
                print(f"    ✓ Solution: Add valid UNSPLASH_API_KEY to .env")
                print(f"    Get key from: https://unsplash.com/oauth/applications")
                return self._get_default_image_urls(search_term)
            else:
                print(f"  ⚠ Unsplash API error: {response.status_code}")
                return self._get_default_image_urls(search_term)
                
        except requests.exceptions.ConnectionError:
            print(f"  ⚠ Cannot connect to Unsplash, using default images")
            search_term = tags.split(',')[0].strip() if tags else 'technology'
            return self._get_default_image_urls(search_term)
        except Exception as e:
            print(f"  ⚠ Error fetching image: {e}")
            search_term = tags.split(',')[0].strip() if tags else 'technology'
            return self._get_default_image_urls(search_term)
    
    def _get_default_image_urls(self, keyword: str = 'technology') -> tuple:
        """
        Generate default Unsplash URLs as fallback.
        """
        keyword = keyword.lower().replace(' ', '-').strip()
        
        # Use Unsplash search with parameters
        full_image_url = f"https://images.unsplash.com/photo-1496307042754-b4aa456c4a2d?q=80&w=1600&auto=format&fit=crop&crop=entropy"
        placeholder_url = f"https://images.unsplash.com/photo-1496307042754-b4aa456c4a2d?q=10&w=40&auto=format&fit=crop"
        
        return (full_image_url, placeholder_url)
    
    def generate_blog_post_with_llm(self, topic: Dict) -> str:
        """Generate blog post content using local LLM."""
        
        # Create a prompt for the LLM
        prompt = f"""Write a comprehensive professional blog post with HTML markup for the following topic:

Title: {topic['title']}
Excerpt: {topic['excerpt']}
Tags: {topic['tags']}

Requirements:
- Write detailed, engaging, and informative content
- MINIMUM 300 words (approximately 2000-2500 characters)
- Use ONLY semantic HTML tags  (p, h2, h3, h4, ul, li, ol, strong, em) for formatting
- Include 3-4 main sections with headers (h2 tags)
- Add 2-3 subsections (h3 tags) within main sections
- Include bullet points <ul><li> or numbered lists <ol><li> where relevant
- Add practical examples, tips, or case studies
- Keep paragraphs concise (2-4 sentences) and scannable
- Include an introduction and conclusion section
- Use strong/em tags for emphasis on key points
- Do NOT include the title in the content
- Content should be well-structured and professional
- No nedd to have html page structure like <html>, <body>, etc.

CRITICAL - DO NOT INCLUDE:
- <!DOCTYPE html>
- <html> tags
- <head> tags
- <body> tags
- <meta> tags
- <title> tags
- Any document structure - only content with HTML tags

Generate ONLY the content starting directly with a semantic tag like <p> or <h2>.
Example format:
<p>Your introduction paragraph here...</p>
<h2>Section Title</h2>
<p>Section content...</p>
<ul>
<li>Bullet point</li>
</ul>"""

        print(f"  Generating content via LLM...")
        llm_content = self.call_local_llm(prompt)
        
        if not llm_content:
            # Fallback to basic content if LLM fails
            llm_content = f"<p>{topic['excerpt']}</p>\n<p>This blog post covers key aspects of {topic['title']}. Please review and expand with more detailed content.</p>"
        
        tags = topic['tags']
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(',') if t.strip()]
        else:
            tags = []
        
        post_id = f"post-{int(datetime.now().timestamp())}"

        image_url, placeholder_url = self.get_fallback_image_url(topic['tags'])
        
        post = {
            'id': post_id,
            'title': topic['title'],
            'author': topic['author'],
            'date': topic['date'],
            'tags': tags,
            'image': image_url,
            'placeholder': placeholder_url,
            'excerpt': topic['excerpt'],
            'content': llm_content
        }
        
        return json.dumps(post, indent=2)
    
    def create_feature_branch(self, branch_name: str) -> bool:
        """Create and checkout a new feature branch."""
        try:
            # Fetch latest from remote
            self.repo.remotes.origin.fetch()
            
            # Create feature branch from base branch
            base_ref = f'origin/{self.base_branch}'
            self.repo.create_head(branch_name, base_ref)
            self.repo.heads[branch_name].checkout()
            
            print(f"✓ Created and checked out branch: {branch_name}")
            return True
        except Exception as e:
            print(f"✗ Error creating feature branch: {e}")
            return False
    
    def add_post_to_posts_js(self, post_content: str) -> bool:
        """Add generated post to posts.js file."""
        try:
            posts_file = self.repo_path / 'posts.js'
            
            # Read existing posts.js
            with open(posts_file, 'r') as f:
                content = f.read()
            
            # Parse the post JSON
            post_json = json.loads(post_content)
            post_js = json.dumps(post_json, indent=4)
            
            # Insert new post at the beginning of the array
            insertion_point = content.find('[') + 1
            updated_content = content[:insertion_point] + '\n    ' + post_js.replace('\n', '\n    ') + ',' + content[insertion_point:]
            
            # Write back to file
            with open(posts_file, 'w') as f:
                f.write(updated_content)
            
            print(f"✓ Added post to posts.js")
            return True
        except Exception as e:
            print(f"✗ Error updating posts.js: {e}")
            return False
    
    def commit_changes(self, post_title: str) -> bool:
        """Commit changes to the feature branch."""
        try:
            self.repo.index.add(['posts.js'])
            commit_message = f"feat: Add blog post - {post_title}"
            self.repo.index.commit(commit_message)
            print(f"✓ Committed changes: {commit_message}")
            return True
        except Exception as e:
            print(f"✗ Error committing changes: {e}")
            return False
    
    def push_to_remote(self, branch_name: str) -> bool:
        """Push feature branch to remote."""
        try:
            self.repo.remotes.origin.push(branch_name)
            print(f"✓ Pushed branch to remote: {branch_name}")
            return True
        except Exception as e:
            print(f"✗ Error pushing to remote: {e}")
            return False
    
    def create_pull_request(self, branch_name: str, post_title: str) -> Optional[str]:
        """Create a pull request on GitHub."""
        try:
            url = f"https://api.github.com/repos/{self.github_owner}/{self.github_repo}/pulls"
            headers = {
                "Authorization": f"token {self.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }
            
            data = {
                "title": f"Add blog post: {post_title}",
                "body": f"Automated blog post from topic: {post_title}\n\nGenerated on: {datetime.now().isoformat()}",
                "head": branch_name,
                "base": self.base_branch
            }
            
            response = requests.post(url, headers=headers, json=data)
            
            if response.status_code == 201:
                pr_data = response.json()
                pr_url = pr_data['html_url']
                print(f"✓ Created pull request: {pr_url}")
                return pr_url
            else:
                print(f"✗ Error creating PR: {response.status_code}")
                print(response.text)
                return None
        except Exception as e:
            print(f"✗ Error creating pull request: {e}")
            return None
    
    def process_topics(self, csv_file: str) -> bool:
        """Main workflow: read CSV topics and create blog posts."""
        try:
            print("Reading topics from CSV file...")
            topics = self.read_topics_from_csv(csv_file)
            
            if not topics:
                print("No topics found matching today's date in CSV file.")
                return False
            
            print(f"Found {len(topics)} topic(s) with today's date\n")
            
            for idx, topic in enumerate(topics, 1):
                print(f"\n{'='*60}")
                print(f"Processing topic {idx}/{len(topics)}: {topic['title']}")
                print(f"{'='*60}")
                
                # Generate blog post with LLM
                post_content = self.generate_blog_post_with_llm(topic)
                
                # Create feature branch
                sanitized_title = topic['title'].lower().replace(' ', '-')[:30]
                branch_name = f"{self.feature_branch_prefix}{sanitized_title}-{int(datetime.now().timestamp())}"
                
                if not self.create_feature_branch(branch_name):
                    continue
                
                # Add post to posts.js
                if not self.add_post_to_posts_js(post_content):
                    continue
                
                # Commit changes
                if not self.commit_changes(topic['title']):
                    continue
                
                # Push to remote
                if not self.push_to_remote(branch_name):
                    continue
                
                # Create PR
                pr_url = self.create_pull_request(branch_name, topic['title'])
                
                if pr_url:
                    print(f"✓ Successfully created PR: {pr_url}\n")
                else:
                    print(f"✗ Failed to create PR for {topic['title']}\n")
            
            return True
        except Exception as e:
            print(f"Error processing topics: {e}")
            return False

def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python blog_automation.py <csv_file>")
        print("Example: python blog_automation.py topics.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    try:
        automation = BlogAutomation()
        automation.process_topics(csv_file)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
