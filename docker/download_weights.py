from huggingface_hub import snapshot_download
import os


current_dir = os.getcwd()
repo_ids = ["IDEA-Research/grounding-dino-tiny", "IDEA-Research/grounding-dino-base"]
for repo_id in repo_ids:
    model_name = repo_id.split("/")[1]
    snapshot_download(repo_id=repo_id, local_dir=f"{current_dir}/{model_name}")
