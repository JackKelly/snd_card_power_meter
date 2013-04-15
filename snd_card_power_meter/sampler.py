"""Code for sampling from the machine's sound card."""

from __future__ import print_function, division
import time, threading
import logging
log = logging.getLogger("scpm")
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
        self._not_reading_audio = None
        self._audio_stream = None
        self._port_audio = None
        self.adc_data_queue = None        
        
    def open(self):
        log.info("Opening Sampler...")
        self._abort = threading.Event()
        self._safe_to_stop_audio_stream = threading.Event()
        self._safe_to_stop_audio_stream.set()
        self.adc_data_queue = Queue()
        self._port_audio = pyaudio.PyAudio()
        log.info("SAMPLE RATE   = {}".format(config.FRAME_RATE))
        log.info("SAMPLE WIDTH  = {}".format(config.SAMPLE_WIDTH))
        self._audio_stream = self._port_audio.open(
                                 format=config.SAMPLE_FORMAT,
                                 channels=config.N_CHANNELS,
                                 rate=config.FRAME_RATE,
                                 input=True,
                                 frames_per_buffer=config.FRAMES_PER_BUFFER)
        log.info("Successfully opened Sampler.")

    def run(self):
        """This will be run as a separate thread."""
        log.info("Running sampler...") 
        while not self._abort.is_set():
            self.adc_data_queue.put(self.get_adc_data())
        
    def get_adc_data(self):
        """
        Get data from the sound card's analogue to digital converter (ADC).
        
        Returns:
            Bunch with fields:
            - time (float): UNIX timestamp immediately prior to sampling
            - frames_list (list)
        """
        t = time.time()
        frames = []
        RETRIES = 5
        for i in range(config.N_READS_PER_QUEUE_ITEM):
            for __ in range(RETRIES):
                if self._abort.is_set():
                    return
                
                data = None
                self._safe_to_stop_audio_stream.clear()
                try:
                    data = self._audio_stream.read(config.FRAMES_PER_BUFFER)
                except IOError, e:
                    self._safe_to_stop_audio_stream.set()
                    log.warn(str(e) + " at iteration {}/{}"
                             .format(i,config.N_READS_PER_QUEUE_ITEM))
                    if data is not None:
                        log.info("read() did return data.  Will use this data")
                        frames.append(data)
                        break
                else:
                    self._safe_to_stop_audio_stream.set()
                    frames.append(data)
                    break
    
        return Bunch(time=t, data=frames, sample_width=config.SAMPLE_WIDTH)
    
    def terminate(self):
        log.info("Terminating Sampler")
        self._abort.set()
        
        self._safe_to_stop_audio_stream.wait()
        
        if self._audio_stream is not None:
            self._audio_stream.stop_stream()
            self._audio_stream.close()

        if self._port_audio is not None:
            self._port_audio.terminate()
