"""Fresh model runs on the saved weekly inputs using the historical source snapshot."""

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--variant", choices=["all", "react", "react-explore", "debate"], default="all"
    )
    args = parser.parse_args()
    source = Path(__file__).resolve().parent / "runs/weekly-20261118"
    sys.path.insert(0, str(source / "source"))
    import eval.run_weekly as frozen

    assert Path(frozen.__file__).resolve().is_relative_to(source / "source")
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((source / "manifest.json").read_text())
    for name in ["manifest.json", "profile.json"]:
        shutil.copyfile(source / name, args.output / name)
    shutil.copytree(source / "fixtures", args.output / "fixtures")
    jobs = []
    variants = frozen.VARIANTS if args.variant == "all" else [args.variant]
    for window in manifest["windows"]:
        week = window["task"]
        (args.output / week).mkdir()
        for name in [
            "corpus.json",
            "rankings.json",
            "e5-details.json",
            "bm25.json",
            "e5.json",
        ]:
            shutil.copyfile(source / week / name, args.output / week / name)
        jobs.extend((window, variant) for variant in variants)
    frozen.save(
        args.output / "replay-config.json",
        dict(
            source=str(source),
            variants=list(variants),
            workers=3,
            note="Fresh model outputs; copied manifest describes historical inputs/source, not replay creation time. New selections may need new human labels.",
        ),
    )
    fixtures = frozen.SharedFixtures(args.output / "fixtures")

    def execute(job):
        window, variant = job
        frozen.run_agent(args.output, manifest, window, variant, fixtures)
        directory = args.output / window["task"] / variant
        for path in directory.glob("*.jsonl"):
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            path.write_text(
                "".join(
                    json.dumps(
                        {**row, "variant": variant, "task": window["task"]},
                        ensure_ascii=False,
                    )
                    + "\n"
                    for row in rows
                )
            )

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(execute, jobs))
    print(f"Finished {len(jobs)} runs. Results: {args.output}")


if __name__ == "__main__":
    main()
