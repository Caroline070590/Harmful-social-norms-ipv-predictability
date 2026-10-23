#!/usr/bin/env python3
"""Check and reproduce the current public aggregate analyses.

The national command refits fixed models. Supplementary replays validate public
aggregate calculations only; they never train on respondent-level records.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile

BASE = Path(__file__).resolve().parent
REPO = BASE.parent


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def local_path(relative):
    path = (BASE / relative).resolve()
    if BASE not in path.parents:
        raise ValueError("Analysis paths must remain within revision_2026: " + relative)
    return path


def module_records():
    return read_json(BASE / "module_index.json")["modules"]


def check():
    manifest = read_json(BASE / "MANIFEST.json")
    for relative, digest in manifest["files"].items():
        path = local_path(relative)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError("Supplied analysis file differs from its manifest: " + relative)
        if path.suffix == ".py":
            compile(path.read_text(encoding="utf-8"), relative, "exec")
    for item in module_records():
        folder = local_path(item["folder"])
        if not (folder / item["replay_script"]).is_file():
            raise FileNotFoundError(folder / item["replay_script"])
    result = dict(status="passed", files_checked=len(manifest["files"]),
                  modules=len(module_records()), scope="File integrity, Python syntax and replay entry points; no fitting or downloads.")
    print(json.dumps(result, indent=2))
    return result


def output_path(value):
    output = value.resolve()
    # Prevent an output option from overwriting supplied analysis inputs/code.
    if output == BASE or BASE in output.parents or output in BASE.parents:
        raise ValueError("Choose an output folder outside revision_2026 (for example reproduced/national).")
    output.mkdir(parents=True, exist_ok=True)
    return output


def run_national(output):
    output = output_path(output)
    subprocess.run([sys.executable, str(BASE / "central/reproduce.py"),
                    "--output-dir", str(output)], cwd=REPO, check=True)


def run_aggregates(output):
    output = output_path(output)
    reports = {}
    with tempfile.TemporaryDirectory(prefix="ipv_public_aggregate_") as temp:
        scratch = Path(temp)
        for item in module_records():
            folder = local_path(item["folder"])
            work = scratch / item["id"]
            shutil.copytree(folder, work)
            command = [sys.executable, str(work / item["replay_script"]), *item.get("replay_args", [])]
            result = subprocess.run(command, cwd=work, capture_output=True, text=True)
            (output / (item["id"] + ".log")).write_text(result.stdout + result.stderr, encoding="utf-8")
            if result.returncode:
                raise RuntimeError("Aggregate replay failed for " + item["id"] + "; see its output log.")
            reports[item["id"]] = {"status": "passed", "scope": item["replay_scope"]}
            print(item["id"] + ": passed", flush=True)
    report = dict(status="passed", modules=reports,
                  scope="Public aggregate calculations and saved-error uncertainty only; no individual data, model refitting or neural retraining.")
    (output / "AGGREGATE_REPLAY.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check", help="Validate supplied files and syntax without executing analysis stages")
    national = commands.add_parser("national", help="Refit the central country/time and external models")
    national.add_argument("--output", type=Path, default=REPO / "reproduced/national")
    aggregates = commands.add_parser("aggregates", help="Replay supplementary aggregate calculations in temporary copies")
    aggregates.add_argument("--output", type=Path, default=REPO / "reproduced/aggregate_checks")
    args = parser.parse_args()
    if args.command == "check":
        check()
    elif args.command == "national":
        run_national(args.output)
    else:
        run_aggregates(args.output)


if __name__ == "__main__":
    main()
