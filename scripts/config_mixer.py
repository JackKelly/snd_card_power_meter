#! /usr/bin/python
"""
Requires the UNIX tool 'amixer'.
"""

from __future__ import print_function
import subprocess, sys

def run_command(cmd):
    """Run a UNIX shell command.
    
    Args:
        cmd (list of strings)
    """
    print("Running '{}'...".format(" ".join(cmd)))
    try:
        proc = subprocess.Popen(cmd, stderr=subprocess.PIPE)
        proc.wait()
    except Exception as e:
        cmd_str = " ".join(cmd)
        print("ERROR: Failed to run {}".format(cmd_str), file=sys.stderr)
        print("ERROR:", str(e), file=sys.stderr)
    else:
        if proc.returncode == 0:
            print("Successfully ran '{}'".format(" ".join(cmd)))
        else:
            print("ERROR: Failed to run '{}'".format(" ".join(cmd)),
                   file=sys.stderr)
            print(proc.stderr.read(), file=sys.stderr)
    print("")


def config_mixer():
    print("Configuring mixer...")
    run_command(["amixer", "sset", "Input Source", "Rear Mic"])
    run_command(["amixer", "set", "Digital", "60", "cap", "capture"])
    run_command(["amixer", "set", "Capture", "15dB,6dB", "cap", "capture"])
    run_command(["amixer", "set", "Rear Mic Boost", "0", "cap", "capture"])
    run_command(["amixer", "set", "Front Mic Boost", "0", "cap", "capture"])


if __name__=="__main__":
    config_mixer()
