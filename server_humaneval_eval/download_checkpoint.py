import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


def main() -> int:
    parser = argparse.ArgumentParser(description="Download the LightLM CPT checkpoint from Hugging Face.")
    parser.add_argument("--repo-id", default="Virg1n/lightlm-ffn-code-cpt")
    parser.add_argument("--filename", default="checkpoints/cpt_step_00037918.pt")
    parser.add_argument("--local-dir", default=".")
    parser.add_argument(
        "--token",
        default=True,
        help="Use HF_TOKEN from the environment by default. Pass a literal token only if needed.",
    )
    args = parser.parse_args()

    path = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.filename,
        local_dir=args.local_dir,
        token=args.token,
    )
    print(Path(path).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
