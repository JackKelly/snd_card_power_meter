"""
Get data from a Watts Up meter over USB.

Requirements:

    - wattsup LinuxUtility on the path
          Download from https://www.wattsupmeters.com/secure/support.php
          
    - wattsup meter plugged into ttyUSB0

Usage:
    wu = WattsUp() 
    wu.open() # connect to WattsUp meter
    data = wu.get() # blocking
"""

from __future__ import print_function, division
import subprocess, shlex, sys, time, atexit, os
from bunch import Bunch

def _parse_wu_line(rawline):
    """Convert a line of text from the Watts Up.
    
    Args:
        rawline (str)
    
    Returns:
        A Bunch with fields:
        - time (int): UNIX timestamp
        - volts (float)
        - amps (float)
        - power_factor (float)
    """

    line = [word.strip(',') for word in rawline.split()]
    
    # Process time (first column)
    try:
        wattsup_time = time.strptime(line[0], "[%H:%M:%S]")
    except ValueError:
        # If we failed to convert the time then this is very unlikely
        # to be a valid line of data.
        print("ERROR reading wattsup line: '{}'"
              .format(rawline), file=sys.stderr)
        return None

    data = Bunch() # what we return

    # The wattsup utility only gives HH:MM:SS so to create a "full"
    # UNIX timestamp we need to merge the HH:MM:SS data from wattsup
    # with the current system date.
    now = time.localtime()            
    t = time.struct_time((now.tm_year, now.tm_mon, now.tm_mday, 
                          wattsup_time.tm_hour,
                          wattsup_time.tm_min,
                          wattsup_time.tm_sec,
                          now.tm_wday, now.tm_yday, now.tm_isdst))
    
    data.time = time.mktime(t) # convert time struct to UNIX timestamp
    data.time = int(data.time) # the wattsup doesn't return sub-second times
    
    # Process volts (second column)
    data.volts = float(line[1])
    
    # Process current (third column)
    data.amps = float(line[2]) / 100
    
    # Power factor (fourth column)
    data.power_factor = float(line[3]) / 10
    
    return data


class WattsUpError(Exception):
    pass


class WattsUp(object):
    """Connects to a WattsUp meter over USB.
    
    Attributes:
        _wu_process (subprocess.Popen): watts up process
    """
    def __init__(self):
        self._wu_process = None
        
    def open(self):
        """Open connection with Watts Up."""
        
        # Check if an old wattsup process is already running
        RETRIES = 5
        for _ in range(RETRIES):
            try:
                pid = subprocess.check_output(['pidof', '-sx', 'wattsup'])
            except subprocess.CalledProcessError:
                break
            else:
                print("WARNING: wattsup is already running.  Attempting to kill it...")
                os.kill(int(pid), 1)
            
        ON_POSIX = 'posix' in sys.builtin_module_names
        CMD = "wattsup -t ttyUSB0 volts amps power-factor"
        self._wu_process = subprocess.Popen(shlex.split(CMD), 
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.STDOUT,
                                            bufsize=4096, close_fds=ON_POSIX)
        
        time.sleep(1) # wait to make sure wattsup stays alive
        self._check_wattsup_is_running()
        atexit.register(self.terminate)
        print("Successfully initialised wattsup")

    def _check_wattsup_is_running(self):
        if self._wu_process is None:
            raise WattsUpError("_wu_process has not been started!")
        else:
            self._wu_process.poll()
            if self._wu_process.returncode is not None:
                # process has terminated so stdout will have an EOF so
                # stdout.read() will return.
                print("wattsup error:", self._wu_process.stdout.read(),
                      file=sys.stderr)
                raise WattsUpError("ERROR: wattsup has died")
        
    def get(self):
        # Check wattsup process is still alive, if not then raise WattsUpError
        self._check_wattsup_is_running()
        line = self._wu_process.stdout.readline()        
        return _parse_wu_line(line)

    def terminate(self):
        if self._wu_process is not None:
            self._wu_process.poll()
            if self._wu_process.returncode is None: # process is still running
                print("Terminating wattsup")
                self._wu_process.terminate()        

    def __del__(self):
        self.terminate()
