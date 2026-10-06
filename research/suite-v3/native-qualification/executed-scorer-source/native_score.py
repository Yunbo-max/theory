"""Native entry point for the unmodified official scorer; CPU only.

This is a separate process from training. EvalPlus supplies per-program resource
limits and its reliability guard; this wrapper does not claim an OS sandbox.
"""
import argparse
import json
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--problems",required=True)
    parser.add_argument("--output",required=True)
    parser.add_argument("--raw")
    parser.add_argument("--qualify",action="store_true")
    args=parser.parse_args()
    os.environ["HUMANEVAL_OVERRIDE_PATH"]=str(Path(args.problems).resolve())
    # The override must be set BEFORE importing the official dataset module.
    from evalplus.data import get_human_eval_plus
    from evalplus.sanitize import sanitize
    from evalplus.evaluate import evaluate
    problems=get_human_eval_plus()
    if args.qualify:
        rows=[{"task_id":k,"solution":v["prompt"]+v["canonical_solution"]} for k,v in problems.items()]
    else:
        if args.raw is None:
            parser.error("--raw is required unless --qualify is selected")
        raw=[json.loads(line) for line in Path(args.raw).read_text().splitlines() if line.strip()]
        rows=[{"task_id":r["task_id"],"solution":sanitize(r["text"],entrypoint=problems[r["task_id"]]["entry_point"])} for r in raw]
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    samples=output/"samples.jsonl"
    samples.write_text("".join(json.dumps(row)+"\n" for row in rows))
    evaluate(dataset="humaneval",samples=str(samples),parallel=2,test_details=True,
             min_time_limit=1,gt_time_limit_factor=4)


if __name__=="__main__":
    main()
