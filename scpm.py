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
  
"""

# TODO:
#  * set mixer levels
#  * do power calcs
#  * sample at 96000, 20-bit
#  * see if wave.writeframes() does proper down-sampling

from __future__ import print_function, division
import numpy as np
import alsaaudio # docs: http://pyalsaaudio.sourceforge.net/libalsaaudio.html
import wave
import matplotlib.pyplot as plt
import subprocess
import sys
import argparse
import audioop

def config_mixer():
    print("Configuring mixer...")
    
    ########## SET INPUT SOURCE ####################
    cmd = ["amixer", "sset", "Input Source", "Rear Mic"]
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
    
    # SET VOLUMES
    try:
        capture = alsaaudio.Mixer("Capture")
        capture.setvolume(70,0)
        capture.setvolume(70,1)
    except alsaaudio.ALSAAudioError, e:
        print("ERROR:", str(e), file=sys.stderr)

    try:    
        digital = alsaaudio.Mixer("Digital")
        digital.setvolume(44,0)
        digital.setvolume(44,1)
    except alsaaudio.ALSAAudioError, e:
        print("ERROR:", str(e), file=sys.stderr)
    

def process_audio():
    inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE)
    inp.setchannels(2)
    print(inp.setperiodsize(512))        
    inp.setrate(96000) # Hz
    inp.setformat(alsaaudio.PCM_FORMAT_S24_LE) # signed 32-bit Little Endian
    
    w = wave.open('test.wav', 'w')
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(44100)
    
    #while True:
    length, data = inp.read()
    print("length=", length)
    a = np.fromstring(data, dtype='int32')
    #print("{:.1f}".format(np.abs(a).mean()))
    plt.plot(a)
    plt.show()
    w.writeframes(data)


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
        
    process_audio()
    

if __name__=="__main__":
    main()
