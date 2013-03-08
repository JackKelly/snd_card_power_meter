"""Code for sampling from the machine's sound card."""

from __future__ import print_function, division
import pyaudio # docs: http://people.csail.mit.edu/hubert/pyaudio/
from threading import Thread
import subprocess
from struct import Struct
try:
    from Queue import Queue, Empty
except ImportError:
    from queue import Queue, Empty # python 3.x

CHUNK = 1024
FORMAT = pyaudio.paInt32
WIDTH = pyaudio.get_sample_size(FORMAT)
CHANNELS = 2
RATE = 96000 #Hz
RECORD_SECONDS = 1

def start_adc_data_queue_and_thread():
    p = pyaudio.PyAudio()
    audio_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                          input=True, frames_per_buffer=CHUNK)

    adc_data_queue = Queue()
    adc_thread = Thread(target=enqueue_adc_data, args=(adc_data_queue,
                                                       audio_stream))
    adc_thread.daemon = True # this thread dies with the program
    adc_thread.start()
    
    return adc_data_queue, adc_thread


def enqueue_adc_data(adc_data_queue, audio_stream):
    """This will be run as a separate thread.""" 
    while True:
        adc_data_queue.put(get_adc_data(audio_stream))


def get_adc_data(audio_stream):
    """
    Get data from the sound card's analogue to digital converter (ADC).
    
    Returns:
        Struct named tuple with fields:
        - time (float): UNIX timestamp immediately prior to sampling
        - data (binary string): stereo ADC data
    """
    t = time.time()
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        for retry in range(5):
            try:
                data = audio_stream.read(CHUNK)
            except IOError, e:
                print("ERROR: ", str(e), file=sys.stderr)
            else:
                frames.append(data)
                break

    stereo = b''.join(frames)
    return Struct(time=t, data=stereo)


def run_command(cmd):
    """Run a UNIX shell command.
    
    Args:
        cmd (list of strings)
    """
    try:
        proc = subprocess.Popen(cmd, stderr=subprocess.PIPE)
        proc.wait()
    except Exception, e:
        print("ERROR: Failed to run '{}'".format(" ".join(cmd)), file=sys.stderr)
        print("ERROR:", str(e), file=sys.stderr)
    else:
        if proc.returncode == 0:
            print("Successfully ran '{}'".format(" ".join(cmd)))
        else:
            print("ERROR: Failed to run '{}'".format(" ".join(cmd)),
                   file=sys.stderr)
            print(proc.stderr.read(), file=sys.stderr)


def config_mixer():
    print("Configuring mixer...")
    run_command(["amixer", "sset", "Input Source", "Rear Mic"])
    run_command(["amixer", "set", "Digital", "60", "capture"])
    run_command(["amixer", "set", "Capture", "16", "capture"])
