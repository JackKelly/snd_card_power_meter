"""Code for sampling from the machine's sound card."""

from __future__ import print_function, division
import subprocess, time, sys
from threading import Thread
import pyaudio # docs: http://people.csail.mit.edu/hubert/pyaudio/
from bunch import Bunch
import config
try:
    from Queue import Queue
except ImportError:
    from queue import Queue # python 3.x


def start():
    p = pyaudio.PyAudio()
    audio_stream = p.open(format=config.SAMPLE_FORMAT,
                          channels=config.N_CHANNELS,
                          rate=config.FRAME_RATE,
                          input=True,
                          frames_per_buffer=config.FRAMES_PER_BUFFER)

    adc_data_queue = Queue()
    adc_thread = Thread(target=enqueue_adc_data, args=(adc_data_queue,
                                                       audio_stream))
    adc_thread.daemon = True # this thread dies with the program
    adc_thread.start()
    return adc_data_queue, audio_stream


def enqueue_adc_data(adc_data_queue, audio_stream):
    """This will be run as a separate thread.""" 
    while True:
        adc_data_queue.put(get_adc_data(audio_stream))


def get_adc_data(audio_stream):
    """
    Get data from the sound card's analogue to digital converter (ADC).
    
    Returns:
        Bunch with fields:
        - time (float): UNIX timestamp immediately prior to sampling
        - data (binary string): stereo ADC data
    """
    t = time.time()
    frames = []
    for _ in range(config.N_READS_PER_QUEUE_ITEM):
        for __ in range(5):
            try:
                data = audio_stream.read(config.FRAMES_PER_BUFFER)
            except IOError, e:
                print("ERROR: ", str(e), file=sys.stderr)
            else:
                frames.append(data)
                break

    stereo = b''.join(frames)
    return Bunch(time=t, data=stereo)
