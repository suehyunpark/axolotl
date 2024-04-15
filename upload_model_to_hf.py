from huggingface_hub import HfApi
from argparse import ArgumentParser


def upload_folder_to_hf(hf_token, folder_path, repo_id, repo_type="model", ignore_patterns=[]):
    api = HfApi(token=hf_token)
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=True, exist_ok=True)
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type=repo_type,
        ignore_patterns=ignore_patterns
    )

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--hf-token", type=str)
    parser.add_argument("--folder-path", type=str)
    parser.add_argument("--repo-id", type=str)
    parser.add_argument("--repo-type", type=str, default="model")
    parser.add_argument("--ignore-patterns", type=str, nargs="*", default=[])
    args = parser.parse_args()
    
    upload_folder_to_hf(args.hf_token, args.folder_path, args.repo_id, args.repo_type, args.ignore_patterns)