import hashlib
import json
import os
from pathlib import Path
import tempfile
import time


class DeadlineReached(Exception):
    pass


class Deadline:
    def __init__(self, end):
        self.end=float(end)
    def remaining(self):
        return max(0.,self.end-time.time())
    def check(self, reserve=15):
        if self.remaining()<=reserve:
            raise DeadlineReached("Persisted run deadline reached")


def atomic_json(path, value):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("w") as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp,path)


def read_json(path):
    return json.loads(Path(path).read_text())


def jsonl(path, rows):
    path=Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp")
    with tmp.open("w") as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+"\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp,path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def stable_seed(*parts):
    return int(hashlib.sha256("|".join(map(str,parts)).encode()).hexdigest()[:15],16)


def object_hash(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":"),allow_nan=False).encode()).hexdigest()


def event(folder, kind, **fields):
    path=Path(folder)/"events.jsonl"
    path.parent.mkdir(parents=True,exist_ok=True)
    row={"time":time.time(),"kind":kind,**fields}
    with path.open("a") as f:
        f.write(json.dumps(row,allow_nan=False)+"\n")
        f.flush()
    print(json.dumps(row,allow_nan=False),flush=True)


def source_manifest(root):
    root=Path(root)
    paths=[*root.glob("recursive_ssd/*.py"),*root.glob("configs/*.json"),
           *root.glob("scripts/*.sh"),*root.glob("scripts/*.py"),
           *root.glob("vendor/research_autopilot/scripts/*.py"),
           *root.glob("vendor/research_autopilot/schemas/*.json"),
           root/"vendor/research_autopilot/VENDOR.json",
           *root.glob("recursive_ssd/templates/*"),root/"pyproject.toml"]
    files={str(p.relative_to(root)):digest(p) for p in sorted(paths) if p.is_file()}
    return {"files":files,"sha256":object_hash(files)}
