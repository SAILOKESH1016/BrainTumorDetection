import os
import sys
import base64
import json
import urllib.request
import urllib.error

def make_request(url, method="GET", headers=None, data=None):
    if headers is None:
        headers = {}
    
    req_data = None
    if data is not None:
        req_data = json.dumps(data).encode('utf-8')
        headers["Content-Type"] = "application/json"
        
    req = urllib.request.Request(url, data=req_data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as response:
            return response.status, json.loads(response.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode('utf-8')
        try:
            return e.code, json.loads(err_msg)
        except:
            return e.code, {"message": err_msg}
    except Exception as e:
        return 0, {"message": str(e)}

def push_project(username, pat):
    repo_name = "BrainTumorDetection"
    headers = {
        "Authorization": f"token {pat}",
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "Antigravity-IDE-Agent"
    }
    
    # 1. Create GitHub Repository
    print(f"Creating repository '{repo_name}' on GitHub for user '{username}'...")
    create_url = "https://api.github.com/user/repos"
    create_data = {
        "name": repo_name,
        "private": False,
        "description": "Production-Style Brain Tumor MRI Classification Dashboard with Automated Image Validation Layer",
        "auto_init": False
    }
    
    status, res = make_request(create_url, method="POST", headers=headers, data=create_data)
    if status == 201:
        print("Repository created successfully!")
    elif status == 422 and "already exists" in str(res):
        print("Repository already exists on your GitHub account. Uploading files to the existing repo...")
    else:
        print(f"Failed to create repository: Status {status}, Error: {res.get('message')}")
        return False
        
    # 2. Get list of files to upload
    files_to_upload = []
    exclude_dirs = {'.venv', '.git', '__pycache__', 'archive', 'data/non_mri', '.ipynb_checkpoints', '.gemini'}
    exclude_files = {'best_resnet50.h5', 'resnet50.h5', 'test_download.jpg'} # Exclude heavy non-champion weights and temp files
    
    for root, dirs, files in os.walk('.'):
        # Exclude directories in-place
        dirs[:] = [d for d in dirs if d not in exclude_dirs]
        for f in files:
            if f in exclude_files:
                continue
            # Skip caches, Jupyter checkpoints, etc.
            path = os.path.join(root, f)
            rel_path = os.path.relpath(path, '.')
            # Format path for GitHub (forward slashes)
            git_path = rel_path.replace('\\', '/')
            files_to_upload.append((rel_path, git_path))
            
    print(f"Found {len(files_to_upload)} files to upload (excluding large datasets and non-champion weights).")
    
    # 3. Upload each file using GitHub Contents API
    success_count = 0
    for local_path, git_path in files_to_upload:
        print(f"Uploading {git_path}...")
        try:
            with open(local_path, 'rb') as f:
                content_bytes = f.read()
            encoded_content = base64.b64encode(content_bytes).decode('utf-8')
            
            # Check if file already exists to get its SHA (required for updating)
            contents_url = f"https://api.github.com/repos/{username}/{repo_name}/contents/{git_path}"
            status, res = make_request(contents_url, method="GET", headers=headers)
            
            sha = None
            if status == 200:
                sha = res.get('sha')
                
            upload_data = {
                "message": f"Upload {git_path} via Antigravity IDE",
                "content": encoded_content
            }
            if sha:
                upload_data["sha"] = sha
                
            status, res = make_request(contents_url, method="PUT", headers=headers, data=upload_data)
            if status in (200, 201):
                success_count += 1
            else:
                print(f"  Failed to upload {git_path}: {res.get('message')}")
        except Exception as e:
            print(f"  Error processing {git_path}: {e}")
            
    print(f"\nUpload complete! Successfully uploaded {success_count}/{len(files_to_upload)} files.")
    return success_count > 0

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python push_to_github.py <pat> [username]")
        sys.exit(1)
        
    if len(sys.argv) == 2:
        pat = sys.argv[1]
        print("Username not specified. Fetching authenticated user from GitHub API using the token...")
        headers = {
            "Authorization": f"token {pat}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Antigravity-IDE-Agent"
        }
        status, user_res = make_request("https://api.github.com/user", method="GET", headers=headers)
        if status == 200 and "login" in user_res:
            username = user_res["login"]
            print(f"Authenticated as GitHub user: {username}")
        else:
            print(f"Failed to fetch GitHub username using the token. Status {status}, Error: {user_res.get('message')}")
            sys.exit(1)
    else:
        username = sys.argv[1]
        pat = sys.argv[2]
        
    success = push_project(username, pat)
    if success:
        github_link = f"https://github.com/{username}/BrainTumorDetection"
        print(f"\nRepository Link: {github_link}")
        # Run resume update
        print("Updating resumes...")
        os.system(f'.venv\\Scripts\\python.exe update_resume.py "{github_link}"')
    else:
        sys.exit(1)
