"""
Get data from a Watts Up meter over USB.

Requirements:

    - wattsup LinuxUtility on the path
          Download from https://www.wattsupmeters.com/secure/support.php
          
    - wattsup meter plugged into ttyUSB0

Usage:
    wu = WattsUp() # connect to WattsUp meter
    data = wu.get_data() # non-blocking
    print(data)

WULine(time=time.struct_time(tm_year=2013, tm_mon=2, tm_mday=22, 
                             tm_hour=17, tm_min=11, tm_sec=11, 
                             tm_wday=4, tm_yday=53, tm_isdst=0),
       volts=227.1,
       amps=0.0)

"""

from __future__ import print_function
import subprocess, shlex, sys, time, collections
from threading import Thread

try:
    from Queue import Queue, Empty
except ImportError:
    from queue import Queue, Empty # python 3.x

def enqueue_output(out, queue):
    """This is run as a separate thread.""" 
    for line in iter(out.readline, b''):
        queue.put(line)
    out.close()

# Create a named tuple for storing time, volts and amps
# WULine = "Watts Up Line"
WULine = collections.namedtuple('WULine', ['time', 'volts', 'amps'])

class WattsUp(object):
    """Connects to a WattsUp meter over USB.  Instantiates a separate thread
    to read data from the WattsUp and places this data into a queue.
    Lines of data can be read in a non-blocking fashion using get_data.
    
    Attributes:
        _p (subprocess.Popen): process
        _q (Queue)
        _t (Thread)
    """
    def __init__(self):
        ON_POSIX = 'posix' in sys.builtin_module_names
        cmd = "wattsup -t ttyUSB0 volts amps"
        self._p = subprocess.Popen(shlex.split(cmd), stdout=subprocess.PIPE,
                             bufsize=4096, close_fds=ON_POSIX)
        self._q = Queue()
        self._t = Thread(target=enqueue_output, args=(self._p.stdout, self._q))
        self._t.daemon = True # thread dies with the program
        self._t.start()
        
    def get_data(self):
        """
        Non-blocking.
        
        Returns:
            if a line of data is available then returns
                a named tuple with the following fields:
                - time (time.struct_time)
                - volts (float)
                - amps (float)
            else if no data is available then returns None.
        """
        try:
            line = self._q.get_nowait()
        except Empty:
            return None
        else:
            line = line.split()
            
            # Process time (first column)
            try:
                raw_wattsup_time = time.strptime(line[0], "[%H:%M:%S]")
            except ValueError:
                # If we failed to convert the time then this is very unlikely
                # to be a valid line of data.
                return None

            now = time.localtime()            
            t = time.struct_time((now.tm_year, now.tm_mon, now.tm_mday, 
                                  raw_wattsup_time.tm_hour,
                                  raw_wattsup_time.tm_min,
                                  raw_wattsup_time.tm_sec,
                                  now.tm_wday, now.tm_yday, now.tm_isdst))
            
            # Process volts (second column)
            volts = float(line[1].strip(','))
            
            # Process current (third column)
            amps = float(line[2])
            
            return WULine(t, volts, amps)

    def __del__(self):
        self._p.terminate()
