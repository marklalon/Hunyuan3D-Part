import argparse
import os
from pathlib import Path
from typing import List, Tuple

from huggingface_hub import hf_hub_download, login, snapshot_download


def _expand(path: str) -> str:
    return str(Path(os.path.expandvars(os.path.expanduser(path))).resolve())


def _status_line(rows: List[Tuple[str, str, str]]) -> None:
    headers = ("Asset", "Status", "Path")
    widths = [len(h) for h in headers]
    for row in rows:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))

    def fmt(values: Tuple[str, str, str]) -> str:
        return " | ".join(values[i].ljust(widths[i]) for i in range(3))

    print("\nDownload Summary")
    print(fmt(headers))
    print("-+-".join("-" * widths[i] for i in range(3)))
    for row in rows:
        print(fmt(row))


def _download_snapshot(repo_id: str, local_dir: str, dry_run: bool) -> Tuple[str, str, str]:
    target = _expand(local_dir)
    if dry_run:
        return (f"snapshot:{repo_id}", "DRY-RUN", target)
    Path(target).mkdir(parents=True, exist_ok=True)
    path = snapshot_download(repo_id=repo_id, local_dir=target)
    return (f"snapshot:{repo_id}", "DONE", _expand(path))


def _download_file(repo_id: str, filename: str, local_dir: str, dry_run: bool, repo_type: str = "model") -> Tuple[str, str, str]:
    target_dir = _expand(local_dir)
    target_file = _expand(os.path.join(target_dir, filename))
    if dry_run:
        return (filename, "DRY-RUN", target_file)
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        local_dir=target_dir,
    )
    return (filename, "DONE", _expand(path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download required Hunyuan3D-Part model assets.")
    parser.add_argument("--hf-token", type=str, default=None, help="Optional Hugging Face token.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without downloading.")
    parser.add_argument(
        "--include-sonata-demo-data",
        action="store_true",
        help="Also download Sonata demo npz files from pointcept/demo dataset repo.",
    )
    parser.add_argument(
        "--models-root",
        type=str,
        default="./models",
        help="Project-local model root directory (default: ./models).",
    )
    parser.add_argument(
        "--xpart-cache",
        type=str,
        default=None,
        help="Destination for XPart model snapshot (default: <models-root>/tencent/Hunyuan3D-Part).",
    )
    parser.add_argument(
        "--p3sam-cache",
        type=str,
        default=None,
        help="Destination for p3sam.safetensors.",
    )
    parser.add_argument(
        "--sonata-ckpt-cache",
        type=str,
        default=None,
        help="Destination for Sonata checkpoints.",
    )
    parser.add_argument(
        "--sonata-data-cache",
        type=str,
        default=None,
        help="Destination for optional Sonata demo data.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    models_root = _expand(args.models_root)
    args.xpart_cache = args.xpart_cache or os.path.join(models_root, "tencent", "Hunyuan3D-Part")
    args.p3sam_cache = args.p3sam_cache or os.path.join(models_root, "p3sam", "weights")
    args.sonata_ckpt_cache = args.sonata_ckpt_cache or os.path.join(models_root, "sonata", "ckpt")
    args.sonata_data_cache = args.sonata_data_cache or os.path.join(models_root, "sonata", "data")

    if args.hf_token and not args.dry_run:
        login(token=args.hf_token, add_to_git_credential=False)

    rows: List[Tuple[str, str, str]] = []
    errors: List[Tuple[str, str]] = []

    tasks = [
        lambda: _download_snapshot("tencent/Hunyuan3D-Part", args.xpart_cache, args.dry_run),
        lambda: _download_file(
            repo_id="tencent/Hunyuan3D-Part",
            filename="p3sam/p3sam.safetensors",
            local_dir=args.p3sam_cache,
            dry_run=args.dry_run,
        ),
        lambda: _download_file(
            repo_id="facebook/sonata",
            filename="sonata.pth",
            local_dir=args.sonata_ckpt_cache,
            dry_run=args.dry_run,
        ),
        lambda: _download_file(
            repo_id="facebook/sonata",
            filename="sonata_small.pth",
            local_dir=args.sonata_ckpt_cache,
            dry_run=args.dry_run,
        ),
        lambda: _download_file(
            repo_id="facebook/sonata",
            filename="sonata_linear_prob_head_sc.pth",
            local_dir=args.sonata_ckpt_cache,
            dry_run=args.dry_run,
        ),
    ]

    if args.include_sonata_demo_data:
        tasks.extend(
            [
                lambda: _download_file(
                    repo_id="pointcept/demo",
                    filename="sample1.npz",
                    local_dir=args.sonata_data_cache,
                    dry_run=args.dry_run,
                    repo_type="dataset",
                ),
                lambda: _download_file(
                    repo_id="pointcept/demo",
                    filename="sample1_high_res.npz",
                    local_dir=args.sonata_data_cache,
                    dry_run=args.dry_run,
                    repo_type="dataset",
                ),
                lambda: _download_file(
                    repo_id="pointcept/demo",
                    filename="sample1_dino.npz",
                    local_dir=args.sonata_data_cache,
                    dry_run=args.dry_run,
                    repo_type="dataset",
                ),
            ]
        )

    for task in tasks:
        try:
            rows.append(task())
        except Exception as error:
            name = getattr(task, "__name__", "task")
            errors.append((name, str(error)))

    _status_line(rows)

    if errors:
        print("\nErrors")
        for name, message in errors:
            print(f"- {name}: {message}")
        raise SystemExit(1)

    print("\nAll requested assets processed successfully.")


if __name__ == "__main__":
    main()
