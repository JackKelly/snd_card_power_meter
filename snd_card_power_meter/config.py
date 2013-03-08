"""Basic config parameters"""

import pyaudio, os

CALIBRATION_FILENAME = os.path.dirname(__file__) + "/../calibration.cfg"
DOWNSAMPLED_RATE = 16000 # Hz (MIT REDD uses 15kHz but 16kHz is a standard
#                              rate and so increases compatibility)
FLAC_FILENAME_PREFIX = "vi" # short for "voltage and current"

# Sampling parameters
FRAMES_PER_BUFFER = 1024
SAMPLE_FORMAT = pyaudio.paInt32
SAMPLE_WIDTH = pyaudio.get_sample_size(SAMPLE_FORMAT)
N_CHANNELS = 2
FRAME_RATE = 96000 #Hz
RECORD_SECONDS = 1
N_READS_PER_QUEUE_ITEM = FRAME_RATE / FRAMES_PER_BUFFER * RECORD_SECONDS
