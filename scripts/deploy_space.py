"""Deploy the demo Space (space/) to Hugging Face on ZeroGPU, with the model repo mounted read-only.

Creating a Gradio Space and hosting ZeroGPU need a Hugging Face PRO (or Team) plan, or a free account older than
30 days. The model (about 52 GB) is mounted as a read-only volume instead of downloaded, since a Space's ephemeral
disk is about 50 GB.

    uv run --no-project --with huggingface_hub python scripts/deploy_space.py --dry-run   # show what would happen
    uv run --no-project --with huggingface_hub python scripts/deploy_space.py             # create or update
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE_ID = "j-raghavan/s1-decision-demo"
MODEL_ID = "j-raghavan/s1-gemma4-26b-decision"
MOUNT = "/models/s1"  # space/app.py reads the model from here when it is mounted
HARDWARE = "zero-a10g"  # ZeroGPU's hardware id; the app asks for the full-GPU size per call (spaces.GPU size=xlarge)
FILES = ["app.py", "s1_space.py", "calibration.json", "examples.json", "requirements.txt", "README.md"]


def bundle(tmp: Path) -> list[Path]:
    for name in FILES:
        shutil.copy2(ROOT / "space" / name, tmp / name)
    shutil.copy2(ROOT / "examples" / "quickstart.py", tmp / "quickstart.py")  # the tested inference code, unchanged
    return sorted(tmp.iterdir())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    from huggingface_hub import HfApi, Volume

    api = HfApi()
    volumes = [Volume(type="model", source=MODEL_ID, mount_path=MOUNT, read_only=True)]
    with tempfile.TemporaryDirectory() as d:
        files = bundle(Path(d))
        print(f"Space {SPACE_ID} (public, Gradio, {HARDWARE}), model {MODEL_ID} mounted at {MOUNT}")
        print("files:", ", ".join(f.name for f in files))
        if args.dry_run:
            return 0
        exists = api.repo_exists(SPACE_ID, repo_type="space")
        if not exists:
            api.create_repo(SPACE_ID, repo_type="space", space_sdk="gradio", private=False,
                            space_hardware=HARDWARE, space_volumes=volumes)
        api.upload_folder(folder_path=d, repo_id=SPACE_ID, repo_type="space",
                          commit_message="Deploy s1 demo from github.com/j-raghavan/s1-decision-model (space/)")
        if exists:  # creation already set these; on updates make sure they are still in place
            runtime = api.get_space_runtime(SPACE_ID)
            if runtime.requested_hardware != HARDWARE and runtime.hardware != HARDWARE:
                api.request_space_hardware(SPACE_ID, hardware=HARDWARE)
            if not any(v.source == MODEL_ID and v.mount_path == MOUNT for v in (runtime.volumes or [])):
                api.set_space_volumes(SPACE_ID, volumes=volumes)
        runtime = api.get_space_runtime(SPACE_ID)
        print(f"stage {runtime.stage}, hardware {runtime.hardware} (requested {runtime.requested_hardware}), "
              f"volumes {[(v.source, v.mount_path) for v in (runtime.volumes or [])]}")
        print(f"https://huggingface.co/spaces/{SPACE_ID}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
