import argparse
import json
import requests
import yaml
import os
from pathlib import Path

# Parse arguments
parser = argparse.ArgumentParser(description="Assign a TA to a PR")
parser.add_argument("--repo", required=True, help="Repository name (e.g., org_name/student_repo)")
parser.add_argument("--pr", type=int, required=True, help="Pull request number")
args = parser.parse_args()

# File paths
tas_file = Path('/static/yaml/tas.yaml')
assignments_file = Path('/static/json/ta_assignment.json')

# Load TA list from YAML
with tas_file.open() as f:
    tas = yaml.safe_load(f)

# Load or initialize assignment tracking
if assignments_file.exists():
    with assignments_file.open() as f:
        assignments = json.load(f)
else:
    assignments = {ta['github']: 0 for ta in tas}

# Determine TA with the fewest assignments
assigned_ta = min(assignments, key=assignments.get)

# Increment assignment count
assignments[assigned_ta] += 1

# Save updated assignments
with assignments_file.open('w') as f:
    json.dump(assignments, f, indent=2)

# Assign reviewer using GitHub API
def assign_reviewer(repo, pr_number, reviewer):
    token = os.getenv('GITHUB_TOKEN')
    headers = {'Authorization': f'token {token}'}
    
    # Add reviewer to PR
    url_review = f'https://api.github.com/repos/{repo}/pulls/{pr_number}/requested_reviewers'
    data_review = {'reviewers': [reviewer]}
    response_review = requests.post(url_review, headers=headers, json=data_review)
    if response_review.status_code == 201:
        print(f"Reviewer {reviewer} successfully assigned to PR #{pr_number} in {repo}.")
    else:
        print(f"Failed to assign reviewer: {response_review.json()}")

    # Add comment to PR
    url_comment = f'https://api.github.com/repos/{repo}/issues/{pr_number}/comments'
    comment_body = {
        "body": f"Your TA for this assignment is @{reviewer}. They will review your submission soon!"
    }
    response_comment = requests.post(url_comment, headers=headers, json=comment_body)
    if response_comment.status_code == 201:
        print(f"Comment successfully added to PR #{pr_number} in {repo}.")
    else:
        print(f"Failed to add comment: {response_comment.json()}")
    
assign_reviewer(args.repo, args.pr, assigned_ta)
print(f"Assigned TA: {assigned_ta}")