#!/usr/bin/env python3
"""Linux command watchdog: independent deadline, parent-death cleanup and locks.

Private runner entry point. It launches no daemon and executes argv without a
shell. Commands must keep descendants in their foreground process group.
"""
import ctypes
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main():
    expected_parent, timeout, result_path, fd_list, command = sys.argv[1:]
    fds = tuple(json.loads(fd_list)); command = json.loads(command)
    child = None; started = time.monotonic(); status = "failed"; exit_code = None; reason = None
    def identity():
        stat = Path("/proc/self/stat").read_text().rsplit(")",1)[1].split()
        return {"pid":int(Path("/proc/self/stat").read_text().split()[0]),"parent_pid":int(stat[1]),"start_ticks":stat[19]}
    stopped = False
    def stop(signum,frame):
        nonlocal stopped
        stopped = True
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGALRM,stop)
    # Unlike preexec_fn, this runs in a fresh interpreter, avoiding fork/thread
    # locks. Check the parent again after prctl to close the parent-death race.
    libc = ctypes.CDLL(None,use_errno=True)
    if libc.prctl(1,signal.SIGTERM,0,0,0) != 0:
        print("RUN_PARENT_DEATH_GUARD_UNAVAILABLE",file=sys.stderr); return 125
    owner = identity()
    if owner["parent_pid"] != int(expected_parent):
        print("RUN_PARENT_IDENTITY_CHANGED",file=sys.stderr); return 125
    signal.setitimer(signal.ITIMER_REAL,float(timeout))
    try:
        child = subprocess.Popen(command,start_new_session=True,shell=False,pass_fds=fds)
        while True:
            if stopped: raise KeyboardInterrupt
            try: exit_code = child.wait(timeout=0.1); break
            except subprocess.TimeoutExpired: continue
        status = "completed" if exit_code == 0 else "failed"
        try: os.killpg(child.pid,0)
        except ProcessLookupError: pass
        else:
            os.killpg(child.pid,signal.SIGKILL)
            status = "interrupted"; reason = "RUN_UNOWNED_PROCESS_GROUP_CHILDREN"
    except KeyboardInterrupt:
        status = "interrupted"; reason = "RUN_OWNER_LOST_OR_HARD_DEADLINE"
        if child is not None:
            try: os.killpg(child.pid,signal.SIGKILL)
            except ProcessLookupError: pass
            exit_code = child.wait()
    except OSError:
        status = "failed"; reason = "COMMAND_EXECUTION_UNAVAILABLE"
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
    result = {"status":status,"exit_code":exit_code,"reason_code":reason,"seconds":time.monotonic()-started,
              "guard_process":owner,"command":command,"cwd":os.getcwd()}
    path = Path(result_path); temp = path.with_name(path.name+".tmp")
    with temp.open("x") as stream:
        json.dump(result,stream,sort_keys=True,allow_nan=False); stream.flush(); os.fsync(stream.fileno())
    os.replace(temp,path)
    return 0


if __name__ == "__main__": raise SystemExit(main())
