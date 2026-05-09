import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def load_json(path: Path) -> Optional[Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def read_jsonl_tail(path: Path, max_bytes: int = 1024 * 1024) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    try:
        with path.open("rb") as f:
            if path.stat().st_size > max_bytes:
                f.seek(-max_bytes, 2)
                data = f.read()
                first_newline = data.find(b"\n")
                if first_newline != -1:
                    data = data[first_newline + 1 :]
            else:
                data = f.read()
    except OSError:
        return []

    rows = []
    for line in data.decode("utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def step_from_path(path: Path) -> Optional[int]:
    match = re.search(r"step_(\d+)", path.name)
    return int(match.group(1)) if match else None


def latest_result_file(run_dir: Path) -> Optional[Path]:
    candidates: List[Tuple[int, Path]] = []
    for path in run_dir.glob("eval_results_step_*.json"):
        step = step_from_path(path)
        if step is not None:
            candidates.append((step, path))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]


def scalar_items(mapping: Dict[str, Any]) -> Iterable[Tuple[str, Any]]:
    for key, value in sorted(mapping.items()):
        if isinstance(value, (str, int, float, bool)) or value is None:
            yield key, value


def print_results(payload: Any) -> None:
    if payload is None:
        print("results: missing or unreadable")
        return
    results = payload.get("results", payload) if isinstance(payload, dict) else payload
    if not isinstance(results, dict):
        print(f"results: {type(results).__name__}")
        return
    print("results:")
    for task, task_result in sorted(results.items()):
        print(f"  {task}:")
        if isinstance(task_result, dict):
            printed = False
            for key, value in scalar_items(task_result):
                print(f"    {key}: {value}")
                printed = True
            if not printed:
                print(f"    keys: {', '.join(sorted(task_result.keys()))}")
        else:
            print(f"    {task_result}")


def tail_text(path: Path, lines: int) -> List[str]:
    if lines <= 0 or not path.exists():
        return []
    try:
        with path.open("rb") as f:
            max_bytes = 128 * 1024
            if path.stat().st_size > max_bytes:
                f.seek(-max_bytes, 2)
            data = f.read()
    except OSError:
        return []
    return data.decode("utf-8", errors="replace").splitlines()[-lines:]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize LightLM BigCode eval outputs.")
    parser.add_argument("--run-dir", default="cpt_runs/lightlm-python-cpt")
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--tail-lines", type=int, default=20)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    eval_log = run_dir / "logs" / "eval.jsonl"
    events = read_jsonl_tail(eval_log)

    result_path: Optional[Path]
    if args.step is None:
        result_path = latest_result_file(run_dir)
        step = step_from_path(result_path) if result_path else None
    else:
        step = args.step
        result_path = run_dir / f"eval_results_step_{step}.json"

    step_events = [row for row in events if step is None or int(row.get("step", -1)) == step]
    last_start = next((row for row in reversed(step_events) if row.get("event") == "bigcode_eval_start"), None)
    last_end = next((row for row in reversed(step_events) if row.get("event") == "bigcode_eval_end"), None)
    last_skip = next((row for row in reversed(step_events) if row.get("event") == "bigcode_eval_skipped"), None)
    latest_event = last_end or last_skip or last_start

    print(f"run_dir: {run_dir}")
    print(f"step: {step if step is not None else 'unknown'}")
    if latest_event:
        print(f"latest_event: {latest_event.get('event')} at {latest_event.get('time')}")
        if "returncode" in latest_event:
            print(f"returncode: {latest_event.get('returncode')}")
        if latest_event.get("reason"):
            print(f"reason: {latest_event.get('reason')}")
        if latest_event.get("metric_output_path"):
            print(f"metric_output_path: {latest_event.get('metric_output_path')}")
        if latest_event.get("stdout_log_file"):
            print(f"stdout_log_file: {latest_event.get('stdout_log_file')}")
    else:
        print("latest_event: none")

    if result_path:
        print(f"result_file: {result_path}")
        print_results(load_json(result_path))
    else:
        print("result_file: none")

    stdout_path_value = None
    if latest_event:
        stdout_path_value = latest_event.get("stdout_log_file")
    if stdout_path_value:
        stdout_path = Path(stdout_path_value)
        if not stdout_path.is_absolute():
            stdout_path = Path.cwd() / stdout_path
        tail = tail_text(stdout_path, args.tail_lines)
        if tail:
            print("stdout_tail:")
            for line in tail:
                print(f"  {line}")


if __name__ == "__main__":
    main()
