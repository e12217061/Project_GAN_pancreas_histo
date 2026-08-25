import os
from github import Github, GithubException
from dotenv import load_dotenv

class GitHubLogger:
    def __init__(self, token, repo_name, issue_number):
        # Authenticate with GitHub
        self.g = Github(token)
        self.repo = self.g.get_repo(repo_name)
        self.issue = self.repo.get_issue(number=issue_number)
        
    def log_epoch(self, epoch, d_loss, g_loss, fid_score=None):
        """Creates a new comment on the issue with the epoch's results."""
        message = f"### 📊 Epoch {epoch}\n"
        message += f"* **Discriminator Loss**: `{d_loss:.4f}`\n"
        message += f"* **Generator Loss**: `{g_loss:.4f}`\n"
        
        if fid_score is not None:
            message += f"* **FID Score**: `{fid_score:.4f}`\n"
            
        # Post the comment via API
        self.issue.create_comment(message)
        print(f"[GitHub] Successfully logged Epoch {epoch} to Issue #{self.issue.number}")

    def commit_file(self, local_file_path, repo_destination_path, commit_message="Upload artifact"):
        """Commits a local file to the GitHub repository."""
        if not os.path.exists(local_file_path):
            print(f"[GitHub] Error: Local file {local_file_path} not found.")
            return

        # Read the file as binary
        with open(local_file_path, 'rb') as f:
            file_content = f.read()

        try:
            # Check if the file already exists in the repo
            repo_file = self.repo.get_contents(repo_destination_path)
            
            # If it exists, update it (requires the file's current SHA)
            self.repo.update_file(
                path=repo_file.path,
                message=commit_message,
                content=file_content,
                sha=repo_file.sha
            )
            print(f"[GitHub] Successfully updated: {repo_destination_path}")
            
        except GithubException as e:
            if e.status == 404:
                # File does not exist, create it
                self.repo.create_file(
                    path=repo_destination_path,
                    message=commit_message,
                    content=file_content
                )
                print(f"[GitHub] Successfully created: {repo_destination_path}")
            else:
                print(f"[GitHub] API Error while committing {repo_destination_path}: {e}")


def main():
    gh_token =  os.environ.get("GITHUB_TOKEN")

    logger = GitHubLogger(token=gh_token, repo_name="e12217061/Project_GAN_pancreas_histo", issue_number=1)
    
    # Log an example epoch
    logger.log_epoch(epoch=-1, d_loss=0.1234, g_loss=0.5678, fid_score=12.34)
    
    # Commit an example file
    local_file_path = "gan_outputs/samples/epoch_0008.png"
    repo_destination_path = "gan_outputs/samples/epoch_0008.png"
    logger.commit_file(local_file_path, repo_destination_path, commit_message="Upload Test")


if __name__ == "__main__":
    main()