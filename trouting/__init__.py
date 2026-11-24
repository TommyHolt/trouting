"""Trouting: an autorouter for KiCad boards."""
__version__ = "0.4.8"


def keep_awake(pid):
    """macOS: no idle sleep while process `pid` runs (a route takes minutes; the
    screen may still turn off). Elsewhere, and without caffeinate, nothing."""
    import shutil
    import subprocess
    import sys
    if sys.platform == "darwin" and shutil.which("caffeinate"):
        try:
            subprocess.Popen(["caffeinate", "-i", "-w", str(pid)], stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except OSError:
            pass
