"""Push hf_space/ to Hugging Face Space. Usage: HF_TOKEN=hf_xxx python deploy_to_hf.py --space your-username/hyperid-ke"""
import argparse, os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

p = argparse.ArgumentParser()
p.add_argument("--space", required=True, help="username/space-name")
p.add_argument("--dir", default="hf_space")
args = p.parse_args()

token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
if not token:
    raise SystemExit("Set HF_TOKEN env var (https://huggingface.co/settings/tokens)")

api = HfApi(token=token)
create_repo(args.space, repo_type="space", space_sdk="docker", exist_ok=True)
api.upload_folder(repo_id=args.space, repo_type="space", folder_path=Path(args.dir), commit_message="Deploy HyperID-KE API")
print(f"Done → https://huggingface.co/spaces/{args.space}")
