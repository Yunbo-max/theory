"""Only runs in the sandbox container; official problems/tests/scoring unchanged."""
import json
import os
from pathlib import Path
import sys

os.environ["HUMANEVAL_OVERRIDE_PATH"]="/input/problems.jsonl"
from evalplus.data import get_human_eval_plus
from evalplus.sanitize import sanitize
from evalplus.evaluate import evaluate

problems=get_human_eval_plus()
if "--qualify" in sys.argv:
    # Canonical solutions are a scorer qualification only; never sent to training.
    rows=[{"task_id":k,"solution":v["prompt"]+v["canonical_solution"]} for k,v in problems.items()]
else:
    raw=[json.loads(line) for line in Path("/input/raw.jsonl").read_text().splitlines()]
    rows=[{"task_id":r["task_id"],"solution":sanitize(r["text"],entrypoint=problems[r["task_id"]]["entry_point"])} for r in raw]
out=Path("/output/samples.jsonl")
out.write_text("".join(json.dumps(row)+"\n" for row in rows))
evaluate(dataset="humaneval",samples=str(out),parallel=2,test_details=True)
