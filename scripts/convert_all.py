#! /usr/bin/python

"""
Converts all remaining WAV files to FLAC
"""

from __future__ import print_function, division
import subprocess, os, time
import snd_card_power_meter.config as config

def main():
    all_files = os.listdir(config.FLAC_DIR)
    wav_files = [f for f in all_files if f.rpartition('.')[-1]=='wav']
    
    if not wav_files:
        print("No wav files in", config.FLAC_DIR)
        return
    
    # check if the last file is currently being updated.  if so then we
    # mustn't touch it.
    wav_files.sort()
    last_wavfile = config.FLAC_DIR + "/" + wav_files[-1]
    size1 = os.path.getsize(last_wavfile)
    time.sleep(2)
    size2 = os.path.getsize(last_wavfile)
    if size2 > size1:
        print("Not processing", last_wavfile, "because it was just updated.")
        wav_files.pop()
    
    wav_files_prefixes = [f.rpartition('.')[0] for f in wav_files]
    
    for f in wav_files_prefixes:
        # Run new sox process to compress wav file                
        # sox is a high quality audio conversion program
        # the "rate -v -L" option forces very high quality
        # rate conversion with linear phase.
        cmd = ("sox --no-dither --show-progress -V3 {filename}.wav --bits 24"
               " --compression 8 {filename}.flac"
               " rate -v -L {downsampled_rate}"
               " && rm {filename}.wav"
               .format(filename=config.FLAC_DIR + "/" + f,
                       downsampled_rate=config.DOWNSAMPLED_RATE))
    
        print(cmd)
        sox_process = subprocess.Popen(cmd, shell=True)
        sox_process.wait()

if __name__ == '__main__':
    main()
