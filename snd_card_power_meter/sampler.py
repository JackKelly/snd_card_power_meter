"""Code for sampling from the machine's sound card."""

from __future__ import print_function, division
import subprocess, time, sys
import threading
import pyaudio # docs: http://people.csail.mit.edu/hubert/pyaudio/
from bunch import Bunch
import config
try:
    from Queue import Queue
except ImportError:
    from queue import Queue # python 3.x


class Sampler(threading.Thread):
    
    def __init__(self):
        threading.Thread.__init__(self)
        self._abort = None
        self._audio_stream = None
        self._port_audio = None
        self.adc_data_queue = None        
        
    def open(self):
        self._abort = threading.Event()
        self.adc_data_queue = Queue()
        self._port_audio = pyaudio.PyAudio()
        self._audio_stream = self._port_audio.open(
                                 format=config.SAMPLE_FORMAT,
                                 channels=config.N_CHANNELS,
                                 rate=config.FRAME_RATE,
                                 input=True,
                                 frames_per_buffer=config.FRAMES_PER_BUFFER)

    def run(self):
        """This will be run as a separate thread.""" 
        while not self._abort.is_set():
            self.adc_data_queue.put(self.get_adc_data())
        
    def get_adc_data(self):
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
                if self._abort.is_set():
                    return
                
                try:
                    data = self._audio_stream.read(config.FRAMES_PER_BUFFER)
                except IOError, e:
                    print("ERROR: ", str(e), file=sys.stderr)
                else:
                    frames.append(data)
                    break
    
        stereo = b''.join(frames)
        return Bunch(time=t, data=stereo)
    
    def terminate(self):
        print("Terminating Sampler")
        self._abort.set()
        
        if self._audio_stream is not None:
            self._audio_stream.stop_stream()
            self._audio_stream.close()

        if self._port_audio is not None:
            self._port_audio.terminate()
