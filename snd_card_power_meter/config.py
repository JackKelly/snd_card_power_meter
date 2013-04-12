"""Basic config parameters"""

from __future__ import print_function, division
import pyaudio, os, sys, time

##########################################################################
# SAMPLING PARAMETERS
FRAME_RATE = 32000 # The sound card sampling rate in Hz
DOWNSAMPLED_RATE = 16000 # Hz (MIT REDD uses 15kHz but 16kHz is a standard
#                              rate and so increases compatibility)
RECORD_SECONDS = 1 # Seconds to record per queue item
N_CHANNELS = 2 # one for voltage, one for current 
FRAMES_PER_BUFFER = 1024
SAMPLE_FORMAT = pyaudio.paInt32
SAMPLE_WIDTH = pyaudio.get_sample_size(SAMPLE_FORMAT)
N_READS_PER_QUEUE_ITEM = int(round(FRAME_RATE / FRAMES_PER_BUFFER
                                   * RECORD_SECONDS))

##########################################################################
# MAINS PARAMETERS
MAINS_HZ = 50
SAMPLES_PER_MAINS_CYCLE = FRAME_RATE / MAINS_HZ
PHASE_DIFF_TOLERANCE = SAMPLES_PER_MAINS_CYCLE / 4
SAMPLES_PER_DEGREE = SAMPLES_PER_MAINS_CYCLE / 360

##########################################################################
# FILENAMES AND DIRECTORIES
BASE_DATA_DIR = os.environ.get("DATA_DIR")
if BASE_DATA_DIR is None:
    print("Please set the $DATA_DIR environment variable.", file=sys.stderr)
    sys.exit(1)

#################################
# Directory for the *.dat files
DATA_DIR = os.path.realpath(BASE_DATA_DIR) + "/high-freq-mains"

if not os.path.isdir(DATA_DIR):
    os.makedirs(DATA_DIR)    

DAT_FILENAME = DATA_DIR + '/mains-{:d}.dat'.format(int(round(time.time())))

#####################################
# Directory and prefix for FLAC files
FLAC_DIR = "/flac"
if not os.path.isdir(FLAC_DIR):
    os.makedirs(FLAC_DIR)    

FLAC_FILENAME_PREFIX = FLAC_DIR + "/vi-" # short for "voltage and current"

####################################
# Misc filenames
LOG_FILENAME = os.path.dirname(__file__) + "/../scpm.log"
CALIBRATION_FILENAME = "/flac/calibration.cfg"
