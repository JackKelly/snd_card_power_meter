"""
Get data from a Watts Up meter over USB.

Requirements:

    - wattsup LinuxUtility on the path
          Download from https://www.wattsupmeters.com/secure/support.php
          
    - wattsup meter plugged into ttyUSB0

Usage:
    wu = WattsUp() # connect to WattsUp meter
    data = wu.get_nowait() # non-blocking


"""

from __future__ import print_function, division
import subprocess, shlex, sys, time, threading
from bunch import Bunch

try:
    from Queue import Queue, Empty
except ImportError:
    from queue import Queue, Empty # python 3.x
    

def _parse_wu_line(line):
    """Convert a line of text from the Watts Up.
    
    Args:
        line (str)
    
    Returns:
        A Bunch with fields:
        - time (float): UNIX timestamp
        - volts (float)
        - amps (float)
        - power_factor (float)
        
    """
    line = [word.strip(',') for word in line.split()]
    
    # Process time (first column)
    try:
        wattsup_time = time.strptime(line[0], "[%H:%M:%S]")
    except ValueError:
        # If we failed to convert the time then this is very unlikely
        # to be a valid line of data.
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
    
    data.time = time.mktime(t) # convert time struct to UNIX timestamp float.
    
    # Process volts (second column)
    data.volts = float(line[1])
    
    # Process current (third column)
    data.amps = float(line[2]) / 100
    
    # Power factor (fourth column)
    data.power_factor = float(line[3]) / 10
    
    return data


class WattsUpError(Exception):
    pass


class WattsUp(threading.Thread):
    """Connects to a WattsUp meter over USB.  Instantiates a separate thread
    to read data from the WattsUp and places this data into a queue.
    Lines of data can be read in a non-blocking fashion using get_nowait.
    
    Attributes:
        _wu_proc (subprocess.Popen): watts up process
        queue (Queue)
        _abort (threading.Event)
    """
    def __init__(self):
        threading.Thread.__init__(self)
        self.daemon = True
        self.queue = None
        self._wu_proc = None
        self._abort = None
        
    def open(self):
        """Open connection with Watts Up."""
        
        ON_POSIX = 'posix' in sys.builtin_module_names
        cmd = "wattsup -t ttyUSB0 volts amps power-factor"
        self._abort = threading.Event()
        self._wu_proc = subprocess.Popen(shlex.split(cmd), 
                                         stdout=subprocess.PIPE,
                                         stderr=subprocess.STDOUT,
                                         bufsize=4096, close_fds=ON_POSIX)
        
        time.sleep(1) # wait to make sure wattsup stays alive
        self._poll_process()
        if self._abort.is_set():
            self._print_wu_errors()
            self.terminate()
            raise WattsUpError("Failed to initialise wattsup")
        else:
            self.queue = Queue()
            print("Successfully initialised wattsup")
        
    def run(self):
        """This will be run as a separate thread."""
        self._poll_process()
        if self._abort.is_set():
            self.terminate()
            return
        
        for line in iter(self._wu_proc.stdout.readline, b''):
            self._poll_process()
            if self._abort.is_set():
                break
            
            # Check for errors
            if (line == "wattsup: [error] Reading final time stamp: Bad address" or
                "Blech. Giving up on read: Bad address" in line):
                print("ERROR:", line, file=sys.stderr)
                continue                

            self.queue.put(_parse_wu_line(line))
                
        self.terminate()

    def _poll_process(self):
        if self._wu_proc is None:
            self._abort.set()
        else:
            self._wu_proc.poll()
            if self._wu_proc.returncode is not None: # wattsup proc has terminated
                print("wattsup has terminated", file=sys.stderr)
                self._abort.set()
            
    def _print_wu_errors(self):
        print("wattsup error:", self._wu_proc.stdout.read(), file=sys.stderr)
        
    def get_last_nowait(self):
        """Return the last (most recent) item on the queue or None."""
        prev_data = None
        while True:
            try:
                data = self.queue.get_nowait()
            except Empty:
                return prev_data
            else:
                prev_data = data

    def terminate(self):
        self._abort.set()
        if self._wu_proc is not None:
            self._wu_proc.poll()
            if self._wu_proc.returncode is None: # process is still running
                print("Terminating wattsup")
                self._wu_proc.terminate()        

    def __del__(self):
        self.terminate()
