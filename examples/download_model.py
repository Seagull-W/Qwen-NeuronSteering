"""Download the complete official model snapshot for subsequent offline use."""

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--output", default="models/Qwen3.5-4B")
    parser.add_argument("--revision", default="main", help="Branch, tag or commit hash")
    args = parser.parse_args()
    destination = Path(args.output).resolve()
    snapshot_download(repo_id=args.model_id, revision=args.revision,
                      local_dir=destination)
    print(f"Model snapshot downloaded to {destination}")


if __name__ == "__main__":
    main()
