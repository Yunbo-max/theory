#!/usr/bin/env python3
"""Save local Codex session events and research decisions without controlling work."""
import argparse
import contextlib
import copy
import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import shlex
import stat
import sys
import tempfile
import time
import uuid
from pathlib import Path

OWNER = "research-autopilot"
EVENTS = ("SessionStart", "SessionEnd", "UserPromptSubmit", "PreToolUse", "PostToolUse",
          "PermissionRequest", "PreCompact", "PostCompact", "SubagentStart", "SubagentStop", "Stop", "Interrupt")
CAPTURE_EVENTS = {"SessionStart", "SessionEnd", "PreCompact", "PostCompact", "SubagentStop", "Stop"}


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _now():
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _project(project):
    root = Path(project).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError("project is not a directory")
    return root


def _mkdir(root, target):
    current = root
    for part in target.relative_to(root).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("record/config directory is a symlink")
        current.mkdir(mode=0o700, exist_ok=True)
        if not current.is_dir():
            raise ValueError("record/config path is not a directory")


def _regular(path):
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise ValueError("record/config file is not a regular file")


def _atomic(path, content):
    _regular(path)
    fd, temporary = tempfile.mkstemp(prefix=".record-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@contextlib.contextmanager
def _lock(folder):
    path = folder / ".record.lock"
    _regular(path)
    fd = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
    try:
        deadline = time.monotonic() + 1.0
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError("session recorder is busy; this callback was not saved")
                time.sleep(0.01)
        yield
    finally:
        os.close(fd)


def _load_hooks(path):
    _regular(path)
    result = json.loads(path.read_text()) if path.exists() else {"hooks": {}}
    if not isinstance(result, dict) or not isinstance(result.get("hooks", {}), dict):
        raise ValueError("existing hooks configuration is invalid")
    result.setdefault("hooks", {})
    for groups in result["hooks"].values():
        if not isinstance(groups, list) or any(not isinstance(group, dict) or
                not isinstance(group.get("hooks"), list) or
                any(not isinstance(handler, dict) for handler in group["hooks"]) for group in groups):
            raise ValueError("existing hooks configuration is invalid")
    return result


def _ours(handler):
    command = handler.get("command")
    if not isinstance(command, str):
        return False
    try:
        parts = shlex.split(command)
    except ValueError:
        return False
    return any(parts[i:i+2] == ["--owner", OWNER] for i in range(len(parts)-1)) and any(
        Path(part).name == "session_records.py" for part in parts)


def _remove_ours(config):
    for event, groups in list(config["hooks"].items()):
        kept = []
        for group in groups:
            handlers = [handler for handler in group["hooks"] if not _ours(handler)]
            if handlers:
                kept.append({**group, "hooks": handlers})
        if kept:
            config["hooks"][event] = kept
        else:
            del config["hooks"][event]


def _enable_toml(text):
    try:
        import tomllib
    except ImportError as failure:
        raise ValueError("enabling hooks requires Python 3.11+; recording callbacks use Python 3.10+") from failure
    parsed = tomllib.loads(text)
    if not isinstance(parsed.get("features", {}), dict):
        raise ValueError("existing features configuration is invalid")
    if parsed.get("features", {}).get("hooks") is True:
        return text
    if "hooks" in parsed.get("features", {}) and parsed["features"]["hooks"] is not False:
        raise ValueError("existing hooks feature is not a boolean")
    expected = copy.deepcopy(parsed)
    expected.setdefault("features", {})["hooks"] = True
    lines = text.splitlines(keepends=True)
    candidates = []
    for i, line in enumerate(lines):
        if re.fullmatch(r"\s*\[features\]\s*(?:#.*)?\n?", line):
            end = next((j for j in range(i+1, len(lines)) if re.match(r"\s*\[", lines[j])), len(lines))
            section = lines[i+1:end]
            matches = [j for j, value in enumerate(section) if re.match(r"\s*hooks\s*=", value)]
            if matches:
                section[matches[0]] = re.sub(r"(hooks\s*=\s*)false\b", r"\g<1>true", section[matches[0]], count=1)
            else:
                section.insert(0, "hooks = true\n")
            header = line if line.endswith("\n") else line + "\n"
            candidates.append("".join(lines[:i] + [header] + section + lines[end:]))
    candidates.append(re.sub(r"(?m)^(\s*features\.hooks\s*=\s*)false\b", r"\g<1>true", text))
    candidates.extend((text.rstrip() + "\n\n[features]\nhooks = true\n", "features.hooks = true\n" + text))
    for candidate in candidates:
        try:
            if tomllib.loads(candidate) == expected:
                return candidate
        except ValueError:
            pass
    raise ValueError("cannot safely update this TOML layout; set features.hooks = true in the client configuration")


def enable(project, codex_home=None):
    root = _project(project)
    home = Path(codex_home or os.environ.get("CODEX_HOME", str(Path.home()/".codex"))).expanduser().resolve()
    config_dir = root / ".codex"
    _mkdir(root, config_dir)
    hooks_path, toml_path = config_dir/"hooks.json", config_dir/"config.toml"
    config = _load_hooks(hooks_path)
    _regular(toml_path)
    previous_toml = toml_path.read_text() if toml_path.exists() else ""
    updated_toml = _enable_toml(previous_toml)
    _remove_ours(config)
    command = shlex.join([sys.executable, str(Path(__file__).resolve()), "hook", "--project", str(root),
                          "--codex-home", str(home), "--owner", OWNER])
    for event in EVENTS:
        config["hooks"].setdefault(event, []).append({"hooks": [{"type": "command", "command": command,
            "timeout": 3 if event in {"SessionEnd", "Interrupt"} else 30}]})
    updated_hooks = (json.dumps(config, ensure_ascii=False, indent=2)+"\n").encode()
    records = root/".research-autopilot/records"
    backups = records/"setup-backups"
    _mkdir(root, backups)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    for path, content in ((hooks_path, updated_hooks), (toml_path, updated_toml.encode())):
        previous = path.read_bytes() if path.exists() else None
        if previous == content:
            continue
        if previous is not None:
            _atomic(backups/(stamp+"-"+path.name), previous)
        _atomic(path, content)
    return {"configuration_written": True, "hooks_path": str(hooks_path), "records_path": str(records),
            "host_hook_trust": "review_in_client", "client_step": "Open /hooks and review/trust these definitions",
            "capture_scope": "this trusted project in clients supporting these hooks; not previous or other chats"}


def disable(project):
    root = _project(project)
    path = root/".codex/hooks.json"
    if path.exists():
        _mkdir(root, path.parent)
        config = _load_hooks(path)
        _remove_ours(config)
        _atomic(path, (json.dumps(config, ensure_ascii=False, indent=2)+"\n").encode())
    return {"recorder_removed": True, "records_retained": True, "other_hooks_unchanged": True}


def _recover_event_tail(path, folder):
    """Retain a torn last write separately before appending another JSON line."""
    _regular(path)
    if not path.exists():
        return None
    fd = os.open(path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "r+b") as handle:
        size = handle.seek(0, os.SEEK_END)
        if not size:
            return None
        handle.seek(size-1)
        if handle.read(1) == b"\n":
            return None
        start, position = 0, size
        while position:
            offset = max(0, position-65536)
            handle.seek(offset)
            block = handle.read(position-offset)
            boundary = block.rfind(b"\n")
            if boundary >= 0:
                start = offset+boundary+1
                break
            position = offset
        handle.seek(start)
        tail = handle.read(size-start)
        try:
            complete = isinstance(json.loads(tail), dict)
        except (ValueError, UnicodeDecodeError):
            complete = False
        if complete:
            handle.seek(0, os.SEEK_END)
            handle.write(b"\n")
            retained = None
        else:
            directory = folder/"interrupted-events"
            _mkdir(folder, directory)
            backup = directory/(uuid.uuid4().hex+".partial")
            _atomic(backup, tail)
            retained = {"path": str(backup), "bytes": len(tail), "sha256": hashlib.sha256(tail).hexdigest()}
            handle.truncate(start)
        handle.flush()
        os.fsync(handle.fileno())
        return retained


def _capture(folder, transcript_path, codex_home):
    source = Path(transcript_path).expanduser().resolve(strict=True)
    home = Path(codex_home or os.environ.get("CODEX_HOME", str(Path.home()/".codex"))).expanduser().resolve()
    roots = [home/"sessions", home/"archived_sessions"]
    if source.suffix != ".jsonl" or not any(not base.is_symlink() and source.is_relative_to(base) for base in roots):
        raise ValueError("transcript is outside the host session directories")
    stream_id = hashlib.sha256(str(source).encode()).hexdigest()[:24]
    directory = folder/"transcripts"/stream_id
    _mkdir(folder, directory)
    cursor_path = directory/"cursor.json"
    _regular(cursor_path)
    cursor = json.loads(cursor_path.read_text()) if cursor_path.exists() else {}
    if not isinstance(cursor, dict) or (cursor_path.exists() and (
            type(cursor.get("offset")) is not int or cursor["offset"] < 0 or
            type(cursor.get("generation")) is not int or cursor["generation"] < 1 or
            not isinstance(cursor.get("identity"), list) or len(cursor["identity"]) != 2 or
            any(type(value) is not int for value in cursor["identity"]) or
            not isinstance(cursor.get("prefix_sha256"), str) or
            not re.fullmatch(r"[0-9a-f]{64}", cursor["prefix_sha256"]))):
        raise ValueError("native transcript cursor is malformed")
    fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(fd, "rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("host transcript is not a regular file")
        identity = [info.st_dev, info.st_ino]
        offset = cursor.get("offset", 0)
        generation = cursor.get("generation", 1)
        prefix = hashlib.sha256()
        reset = bool(cursor) and (cursor.get("identity") != identity or offset > info.st_size)
        if not reset:
            remaining = offset
            while remaining:
                chunk = handle.read(min(1024*1024, remaining))
                if not chunk:
                    raise ValueError("host transcript changed during capture")
                prefix.update(chunk)
                remaining -= len(chunk)
            reset = bool(cursor) and prefix.hexdigest() != cursor.get("prefix_sha256")
        if reset:
            offset, generation = 0, generation+1
            prefix = hashlib.sha256()
            handle.seek(0)
        capture = {"source_path": str(source), "stream_id": stream_id, "generation": generation,
                   "offset_start": offset, "offset_end": info.st_size, "status": "unchanged"}
        if offset == info.st_size:
            return capture, None
        name = "%012d-%012d-%s.segment" % (offset, info.st_size, uuid.uuid4().hex)
        segment = directory/name
        output_fd = os.open(segment, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        segment_hash = hashlib.sha256()
        try:
            with os.fdopen(output_fd, "wb") as output:
                remaining = info.st_size-offset
                while remaining:
                    chunk = handle.read(min(1024*1024, remaining))
                    if not chunk:
                        raise ValueError("host transcript changed during capture")
                    output.write(chunk)
                    segment_hash.update(chunk)
                    prefix.update(chunk)
                    remaining -= len(chunk)
                output.flush()
                os.fsync(output.fileno())
        except Exception:
            segment.unlink(missing_ok=True)
            raise
        capture.update(status="copied", segment_path=str(segment), sha256=segment_hash.hexdigest(),
                       captured_bytes=info.st_size-offset)
        next_cursor = {"identity": identity, "generation": generation, "offset": info.st_size,
                       "prefix_sha256": prefix.hexdigest()}
        return capture, (cursor_path, next_cursor)


def _record(project, payload, source, event_name, session_id, codex_home=None):
    root = _project(project)
    key = hashlib.sha256(session_id.encode()).hexdigest()[:32] if isinstance(session_id, str) else "unidentified"
    folder = root/".research-autopilot/records/sessions"/key
    _mkdir(root, folder)
    path = folder/"events.jsonl"
    record = {"schema_id": "session-record", "schema_version": "1.0.0", "record_id": uuid.uuid4().hex,
              "recorded_at": _now(), "project_path": str(root), "source": source, "event_name": event_name,
              "session_id": session_id if isinstance(session_id, str) else None,
              "payload": payload, "transcript_captures": []}
    # Validate JSON before file mutation. This recorder does not validate scientific claims.
    _json(record)
    with _lock(folder):
        updates = []
        if source == "codex_hook" and event_name in CAPTURE_EVENTS:
            paths = list(dict.fromkeys(p for p in (payload.get("transcript_path"),
                                      payload.get("agent_transcript_path")) if isinstance(p, str) and p))
            if not paths:
                record["transcript_captures"].append({"status": "unavailable", "reason": "no_host_path"})
            for raw_path in paths:
                try:
                    capture, update = _capture(folder, raw_path, codex_home)
                    record["transcript_captures"].append(capture)
                    if update:
                        updates.append(update)
                except (OSError, ValueError, TypeError):
                    record["transcript_captures"].append({"source_path": raw_path, "status": "unavailable"})
        retained = _recover_event_tail(path, folder)
        if retained:
            record["recovered_event_tail"] = retained
        fd = os.open(path, os.O_CREAT | os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0), 0o600)
        with os.fdopen(fd, "ab") as handle:
            handle.write((_json(record)+"\n").encode())
            handle.flush()
            os.fsync(handle.fileno())
        for cursor_path, cursor in updates:
            _atomic(cursor_path, (_json(cursor)+"\n").encode())
    return {"record_id": record["record_id"], "events_path": str(path), "recorded": True}


def record_hook(project, payload, codex_home=None):
    if not isinstance(payload, dict) or not isinstance(payload.get("hook_event_name"), str):
        raise ValueError("host hook input is unavailable or malformed")
    return _record(project, payload, "codex_hook", payload["hook_event_name"], payload.get("session_id"), codex_home)


def record_decision(project, note, session_id=None):
    if not isinstance(note, dict) or any(not isinstance(note.get(key), str) or not note[key].strip()
                                         for key in ("decision", "rationale")):
        raise ValueError("decision and concise rationale are required")
    return _record(project, note, "worker_decision_note", "ResearchDecision", session_id)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("enable", "disable", "hook", "decision", "where"):
        child = commands.add_parser(name)
        child.add_argument("--project", required=True)
        if name in {"enable", "hook"}:
            child.add_argument("--codex-home")
        if name == "hook":
            child.add_argument("--owner", default=OWNER, choices=[OWNER])
        if name == "decision":
            child.add_argument("--note", required=True, help="actual decision JSON, or - for stdin")
            child.add_argument("--session-id")
    args = parser.parse_args(argv)
    try:
        if args.command == "hook":
            record_hook(args.project, json.load(sys.stdin), args.codex_home)
            result = {}
        elif args.command == "enable":
            result = enable(args.project, args.codex_home)
        elif args.command == "disable":
            result = disable(args.project)
        elif args.command == "decision":
            note = json.load(sys.stdin) if args.note == "-" else json.loads(Path(args.note).read_text())
            result = record_decision(args.project, note, args.session_id)
        else:
            root = _project(args.project)
            result = {"records_path": str(root/".research-autopilot/records"),
                      "host_hook_trust": "check_in_client_/hooks"}
        print(_json(result))
        return 0
    except (OSError, ValueError, TypeError, TimeoutError) as failure:
        if args.command == "hook":
            # Observational only: never deny, approve, rewrite or continue an agent turn.
            print(_json({"systemMessage": "Research session recording failed; this callback was not fully saved. "
                         + type(failure).__name__}))
            return 0
        print(_json({"error": str(failure)}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
