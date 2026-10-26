"""Download the complete official model snapshot for subsequent offline use."""

import argparse
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--output", default="models/Qwen3.5-4B")
    parser.add_argument("--revision", default="main", help="Branch, tag or commit hash")
    parser.add_argument("--endpoint", default="https://hf-mirror.com",
                        help="Model download endpoint (default: domestic HF mirror)")
    args = parser.parse_args()
    # Configure before importing the Hub so its constants use this endpoint.
    os.environ["HF_ENDPOINT"] = args.endpoint
    # Use ordinary HTTP downloads through the mirror rather than the Xet CDN.
    if args.endpoint.rstrip("/") == "https://hf-mirror.com":
        os.environ["HF_HUB_DISABLE_XET"] = "1"
    from huggingface_hub import snapshot_download

    destination = Path(args.output).resolve()
    snapshot_download(repo_id=args.model_id, revision=args.revision,
                      local_dir=destination, endpoint=args.endpoint)
    print(f"Model snapshot downloaded to {destination}")


if __name__ == "__main__":
    main()
