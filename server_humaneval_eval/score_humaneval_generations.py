import argparse
import json
import os
import sys
import types
from pathlib import Path
from typing import Any


def jsonable(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Score saved BigCode HumanEval or MBPP generations with the official harness metric."
    )
    parser.add_argument("--task", choices=("humaneval", "mbpp"), default="humaneval")
    parser.add_argument("--generations-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--harness-dir", default="bigcode-evaluation-harness")
    parser.add_argument("--limit-start", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=3.0)
    parser.add_argument("--allow-code-execution", action="store_true")
    args = parser.parse_args()

    if args.allow_code_execution:
        os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    harness_dir = Path(args.harness_dir).resolve()
    sys.path.insert(0, str(harness_dir))

    tasks_dir = harness_dir / "bigcode_eval" / "tasks"
    custom_metrics_dir = tasks_dir / "custom_metrics"
    tasks_pkg = types.ModuleType("bigcode_eval.tasks")
    tasks_pkg.__path__ = [str(tasks_dir)]
    custom_metrics_pkg = types.ModuleType("bigcode_eval.tasks.custom_metrics")
    custom_metrics_pkg.__path__ = [str(custom_metrics_dir)]
    sys.modules.setdefault("bigcode_eval.tasks", tasks_pkg)
    sys.modules.setdefault("bigcode_eval.tasks.custom_metrics", custom_metrics_pkg)

    if args.task == "humaneval":
        from bigcode_eval.tasks.humaneval import create_task

        task = create_task(True)(num_workers=args.num_workers, timeout=args.timeout)
    else:
        from bigcode_eval.tasks.mbpp import MBPP

        task = MBPP()
        task.num_workers = args.num_workers
        task.timeout = args.timeout

    with Path(args.generations_path).open("r", encoding="utf-8") as f:
        generations = json.load(f)

    dataset = task.get_dataset()
    references = [
        task.get_reference(dataset[index])
        for index in range(args.limit_start, args.limit_start + len(generations))
    ]

    results = {"results": {args.task: jsonable(task.process_results(generations, references))}}
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
