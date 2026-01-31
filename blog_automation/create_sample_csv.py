#!/usr/bin/env python3
"""
Sample CSV file generator for testing the blog automation tool.
"""

import csv
from datetime import datetime, timedelta

def create_sample_csv(filename='sample_topics.csv'):
    """Create a sample CSV file with blog post topics."""
    
    # Get today's date and tomorrow's date properly
    today = datetime.now().strftime('%Y-%m-%d')
    tomorrow = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
    
    sample_topics = [
        {
            'title': 'Getting Started with Docker',
            'excerpt': 'Learn how to use Docker to containerize your applications and simplify deployment.',
            'tags': 'docker,devops,containers',
            'author': 'Saravana Kumar',
            'date': today  # Today's date - will be processed
        },
        {
            'title': 'Advanced Git Workflows for Teams',
            'excerpt': 'Master branching strategies and collaborative workflows with Git for better team coordination.',
            'tags': 'git,collaboration,workflow',
            'author': 'Saravana Kumar',
            'date': today  # Today's date - will be processed
        },
        {
            'title': 'Performance Optimization in Python',
            'excerpt': 'Tips and tricks to make your Python applications faster and more efficient.',
            'tags': 'python,performance,optimization',
            'author': 'Saravana Kumar',
            'date': tomorrow  # Tomorrow's date - will be skipped
        },
        {
            'title': 'Understanding Microservices Architecture',
            'excerpt': 'A comprehensive guide to building, deploying, and managing microservices.',
            'tags': 'architecture,microservices,design',
            'author': 'Saravana Kumar',
            'date': tomorrow  # Tomorrow's date - will be skipped
        }
    ]
    
    # Write CSV file
    try:
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['title', 'excerpt', 'tags', 'author', 'date'])
            writer.writeheader()
            writer.writerows(sample_topics)
        
        print(f"✓ Created sample CSV file: {filename}")
        print(f"  Contains {len(sample_topics)} sample topics")
        print(f"\nColumns: title, excerpt, tags, author, date")
        print(f"\nTopics with today's date ({today}): 2 (will be processed)")
        print(f"Topics with future dates ({tomorrow}): 2 (will be skipped)")
        print(f"\nTo test:")
        print(f"  python blog_automation.py {filename}")
        
    except Exception as e:
        print(f"✗ Error creating CSV file: {e}")

if __name__ == '__main__':
    create_sample_csv()
