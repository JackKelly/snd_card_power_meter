"""Basic config parameters"""

from __future__ import division
import pyaudio, os

CALIBRATION_FILENAME = os.path.dirname(__file__) + "/../calibration.cfg"
DOWNSAMPLED_RATE = 16000 # Hz (MIT REDD uses 15kHz but 16kHz is a standard
#                              rate and so increases compatibility)
FLAC_FILENAME_PREFIX = "vi-" # short for "voltage and current"
DATA_FILENAME = os.path.dirname(__file__) + "/../mains.dat"
LOG_FILENAME = os.path.dirname(__file__) + "/../scpm.log"

# Sampling parameters
FRAMES_PER_BUFFER = 1024
SAMPLE_FORMAT = pyaudio.paInt32
SAMPLE_WIDTH = pyaudio.get_sample_size(SAMPLE_FORMAT)
N_CHANNELS = 2
FRAME_RATE = 48000 #Hz
RECORD_SECONDS = 1
N_READS_PER_QUEUE_ITEM = int(round((FRAME_RATE / FRAMES_PER_BUFFER) 
                                   * RECORD_SECONDS))

MAINS_HZ = 50
SAMPLES_PER_MAINS_CYCLE = FRAME_RATE / MAINS_HZ
PHASE_DIFF_TOLERANCE = SAMPLES_PER_MAINS_CYCLE / 4
SAMPLES_PER_DEGREE = SAMPLES_PER_MAINS_CYCLE / 360
