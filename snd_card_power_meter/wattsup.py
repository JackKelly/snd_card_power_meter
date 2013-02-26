"""
Get data from a Watts Up meter over USB.

Requirements:

    - wattsup LinuxUtility on the path
          Download from https://www.wattsupmeters.com/secure/support.php
          
    - wattsup meter plugged into ttyUSB0

Usage:
    wu = WattsUp() # connect to WattsUp meter
    data = wu.get_nowait() # non-blocking
    print(data)

WULine(time=1361803002.0, volts=230.1, amps=2.4)

"""

from __future__ import print_function, division
import subprocess, shlex, sys, time, collections
from threading import Thread, Event

try:
    from Queue import Queue, Empty
except ImportError:
    from queue import Queue, Empty # python 3.x

def enqueue_output(stdout, stderr, queue, abort_event):
    """This is run as a separate thread.
    
    Args:
        - stdout, stderr: file objects
        - queue (Queue)
        - abort_event (threading.Event)
    """ 
    for line in iter(stdout.readline, b''):
        queue.put(line)
    for err_line in iter(stderr.readline, b''):
        if ("Blech. Giving up on read: Bad address" in err_line or
            "Reading final time stamp: Bad address" in err_line):
            
            print("ERROR:", err_line, file=sys.stderr)
            abort_event.set()
        
    stdout.close()

# Create a named tuple for storing time, volts and amps
# WULine = "Watts Up Line"
WULine = collections.namedtuple('WULine', ['time', 'volts', 'amps'])

def _line_to_tuple(line):
    """Convert a line of text from the Watts Up to a WULine named tuple."""
    line = line.split()
    
    # Process time (first column)
    try:
        wattsup_time = time.strptime(line[0], "[%H:%M:%S]")
    except ValueError:
        # If we failed to convert the time then this is very unlikely
        # to be a valid line of data.
        return None

    # The wattsup utility only gives HH:MM:SS so to create a "full"
    # UNIX timestamp we need to merge the HH:MM:SS data from wattsup
    # with the current date.
    now = time.localtime()            
    t = time.struct_time((now.tm_year, now.tm_mon, now.tm_mday, 
                          wattsup_time.tm_hour,
                          wattsup_time.tm_min,
                          wattsup_time.tm_sec,
                          now.tm_wday, now.tm_yday, now.tm_isdst))
    t = time.mktime(t) # convert time struct to UNIX timestamp float.
    
    # Process volts (second column)
    volts = float(line[1].strip(','))
    
    # Process current (third column)
    amps = float(line[2]) / 100
    
    return WULine(t, volts, amps)

class WattsUp(object):
    """Connects to a WattsUp meter over USB.  Instantiates a separate thread
    to read data from the WattsUp and places this data into a queue.
    Lines of data can be read in a non-blocking fashion using get_nowait.
    
    Attributes:
        _p (subprocess.Popen): process
        _q (Queue)
        _t (Thread)
        _about (threading.Event)
    """
    def __init__(self):
        ON_POSIX = 'posix' in sys.builtin_module_names
        cmd = "wattsup -t ttyUSB0 volts amps"
        self._abort = Event()
        self._p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   bufsize=4096, close_fds=ON_POSIX)
        self._q = Queue()
        self._t = Thread(target=enqueue_output, args=(self._p.stdout,
                                                      self._p.stderr,
                                                      self._q,
                                                      self._abort))
        self._t.daemon = True # thread dies with the program
        self._t.start()
        
    def _check_abort(self):
        if self._abort.is_set():
            print("QUIT", file=sys.stderr)
            sys.exit(1)
        
    def get(self):
        """Blocking"""
        self._check_abort()
        line = self._q.get()
        return _line_to_tuple(line)
        
    def get_nowait(self):
        """
        Non-blocking.
        
        Returns:
            if a line of data is available then returns
                a named tuple with the following fields:
                - time (float): UNIX timestamp
                - volts (float)
                - amps (float)
            else if no data is available then returns None.
        """
        self._check_abort()
        try:
            line = self._q.get_nowait()
        except Empty:
            return None
        else:
            return _line_to_tuple(line) 
        
    def get_last_nowait(self):
        """Return the last (most recent) item on the queue."""
        self._check_abort()
        prev_data = self.get_nowait()        
        while True:
            data = self.get_nowait()
            if data is None:
                return prev_data
            else:
                prev_data = data

    def __del__(self):
        self._p.terminate()
