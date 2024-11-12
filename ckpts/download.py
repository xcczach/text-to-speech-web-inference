from huggingface_hub import snapshot_download
import os
repo_id="coqui/XTTS-v2"
local_dir="./xttsv2"
if not os.path.exists(local_dir):
    os.makedirs(local_dir)
snapshot_download(repo_id, local_dir=local_dir)