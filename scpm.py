#! /usr/bin/python

""" 
THIS CODE IS JUST ME LEARNING TO USE ALSAAUDIO.
IT DOESN'T YET DO ANY POWER MEASUREMENT STUFF! 

Requirements
------------

  Ubuntu packages
     sudo apt-get install python-dev python-pip alsa alsa-tools libasound2-dev

  pyalsaaudio
     * Install on Linux using "sudo pip install pyalsaaudio"
                           or "sudo easy_install pyalsaaudio".

  On Linux, add yourself to the audio users:
     sudo adduser USERNAME audio
     (then log out and log back in for this change to take effect)
  
Calibration with a Watts Up
---------------------------

Log data from Watts Up using
   screen -dmL ./wattsup ttyUSB0 volts
  
"""

# TODO:
#  * set mixer levels
#  * do power calcs
#  * sample at 96000, 20-bit
#  * see if wave.writeframes() does proper down-sampling

from __future__ import print_function, division
import numpy as np
import alsaaudio # docs: http://pyalsaaudio.sourceforge.net/libalsaaudio.html
import pyaudio # docs: http://people.csail.mit.edu/hubert/pyaudio/
import wave
import matplotlib.pyplot as plt
import subprocess
import sys
import argparse
import audioop # docs: http://docs.python.org/2/library/audioop.html

CHUNK = 1024
FORMAT = pyaudio.paInt32
CHANNELS = 1
RATE = 96000
RECORD_SECONDS = 1
WAVE_OUTPUT_FILENAME = "output.wav"
VOLTS_PER_ADC_STEP = 1.07731983340487E-06


p = pyaudio.PyAudio()

def run_command(cmd):
    """Run a UNIX shell command.
    
    Args:
        cmd (list of strings)
    """
    try:
        p = subprocess.Popen(cmd, stderr=subprocess.PIPE)
        p.wait()
    except Exception, e:
        print("ERROR: Failed to run '{}'".format(" ".join(cmd)), file=sys.stderr)
        print("ERROR:", str(e), file=sys.stderr)
    else:
        if p.returncode == 0:
            print("Successfully ran '{}'".format(" ".join(cmd)))
        else:
            print("ERROR: Failed to run '{}'".format(" ".join(cmd)), file=sys.stderr)
            print(p.stderr.read(), file=sys.stderr)


def config_mixer():
    print("Configuring mixer...")
    run_command(["amixer", "sset", "Input Source", "Rear Mic"])
    run_command(["amixer", "set", "Digital", "60", "capture"])
    run_command(["amixer", "set", "Capture", "16", "capture"])
    

def setup_audio_stream():        
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    return stream


def process_audio(stream):
    frames = []
    
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        try:
            data = stream.read(CHUNK)
        except IOError, e:
            print("ERROR: ", str(e), file=sys.stderr)
            # TODO: should we do i-=1 ?
        else:
            frames.append(data)
        
    binary_string = b''.join(frames) 

    matrix = np.fromstring(binary_string, dtype='int32')
    print(audioop.rms(binary_string, p.get_sample_size(FORMAT)))
    sys.stdout.flush()
    print("mean = {:4.1f}v, rms = {:4.1f}v".format(
                                matrix.mean() * VOLTS_PER_ADC_STEP,
                                audioop.rms(binary_string, p.get_sample_size(FORMAT)) * VOLTS_PER_ADC_STEP ))
    # plt.plot(matrix)
    # plt.show()


def setup_argparser():
    # Process command line _args
    parser = argparse.ArgumentParser(description="Record voltage and current"
                                     "  waveforms using the sound card.")
       
    parser.add_argument("--no-mixer-config", dest="config_mixer", 
                        action="store_false",
                        help="Don't modify ALSA mixer (i.e. use existing system settings)")
    
    args = parser.parse_args()

    return args

def main():
    args = setup_argparser()
    
    if args.config_mixer:
        config_mixer()

    stream = setup_audio_stream()
    while True:
        try:
            process_audio(stream)
        except KeyboardInterrupt:
            stream.stop_stream()
            stream.close()
            p.terminate()
            break
        
    print("")

if __name__=="__main__":
    main()
