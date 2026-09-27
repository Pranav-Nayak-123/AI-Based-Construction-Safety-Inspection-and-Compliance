from __future__ import annotations

import argparse
import sys
from pathlib import Path

from app.window import launch_window
from jobs.pipeline_job import FakePipelineJob


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="safety-twin",
        description="Offline construction-safety video analysis with a 2.5D situational twin.",
    )
    parser.add_argument("--version", action="version", version="0.1.0")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="Print scaffold status")
    sub.add_parser("app", help="Open the local Site Twin desk in the browser")

    process = sub.add_parser(
        "process",
        help="Process a fixed-camera clip into a run bundle (side-by-side MP4 + incidents)",
    )
    process.add_argument("video", type=Path, help="Path to an input video file")
    process.add_argument(
        "--site",
        type=Path,
        default=None,
        help="Site config JSON with restricted zones and edges for this camera (R3/R5)",
    )
    process.add_argument(
        "--synthetic",
        action="store_true",
        help="Run the Week-1 synthetic pipeline instead (no models needed; placeholder video)",
    )
    process.add_argument(
        "--output-root",
        type=Path,
        default=Path("output"),
        help="Directory for run folders (default: ./output)",
    )
    process.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Optional fixed run id (default: random)",
    )
    process.add_argument(
        "--skip-video",
        action="store_true",
        help="With --synthetic: skip the placeholder encode (schema-only smoke tests)",
    )

    args = parser.parse_args(argv)

    if args.command in (None, "status"):
        root = Path(__file__).resolve().parents[1]
        print("construction-safety-twin 0.1.0")
        print(f"root={root}")
        print("Commands: status | app | process <video>")
        print("See docs/plan/construction-safety-2.5d-twin-plan-v7.md")
        return 0

    if args.command == "app":
        return launch_window()

    if args.command == "process" and not (args.synthetic or args.skip_video):
        return _process(args)

    if args.command == "process":
        job = FakePipelineJob(
            args.video,
            output_root=args.output_root,
            run_id=args.run_id,
            skip_video=args.skip_video,
        )
        manifest = job.run()
        print(f"run_id={manifest.run_id}")
        print(f"status={manifest.status.value}")
        print(f"output={job.run_dir}")
        print(f"video={job.run_dir / 'safety_twin.mp4'}")
        print(f"report={job.run_dir / 'report.html'}")
        print(manifest.disclaimer)
        return 0

    return 1


def _process(args: argparse.Namespace) -> int:
    # Imported here so `status`, `app` and `--synthetic` work without the vision extra.
    from jobs.process_job import ProcessJob
    from shared.errors import PipelineError

    job = ProcessJob(
        args.video,
        output_root=args.output_root,
        site_config=args.site,
        run_id=args.run_id,
        report=lambda message: print(message, flush=True),
    )
    try:
        manifest = job.run()
    except PipelineError as error:
        print(f"failed: {error}", file=sys.stderr)
        return 2
    print(f"run_id={manifest.run_id}")
    print(f"output={job.run_dir}")
    for warning in manifest.warnings:
        print(f"warning: {warning}")
    print(manifest.disclaimer)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
