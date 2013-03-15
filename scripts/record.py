#! /usr/bin/python

from __future__ import print_function, division
import datetime, subprocess, sys, os, atexit
import snd_card_power_meter.scpm as scpm
from snd_card_power_meter.sampler import Sampler
import snd_card_power_meter.config as config

class Recorder(object):
    
    def __init__(self):
        self.wavfile = None
        self.sox_process = None # a subprocess.Popen object    
        self.sampler = Sampler()

    def start(self):
        cmd = ""
        calibration = scpm.load_calibration_file()
        self.sampler.open()
        self.sampler.start()
        while True:
            adc_data = self.sampler.adc_data_queue.get()
            if adc_data is None:
                print("adc_data was None!", file=sys.stderr)
                continue
            
            split_adc_data = scpm.split_channels(adc_data.data)
            
            # Calculate real power, apparent power, v_rms and write to disk
            adc_rms = scpm.calculate_adc_rms(split_adc_data)
            power = scpm.calculate_calibrated_power(split_adc_data, adc_rms, 
                                                    calibration)

            with open(config.DATA_FILENAME, 'a') as data_file:
                data_file.write('{:.1f} {:.2f} {:.2f} {:.2f}\n'
                                .format(adc_data.time, power.real_power,
                                        power.apparent_power, power.v_rms))
    
            # Check if it's time to create a new FLAC file
            t = datetime.datetime.fromtimestamp(adc_data.time)
            if self.wavfile is None or t.minute == (prev.minute+1)%60:
                # TODO: change to hourly recordings after testing!
                if self.wavfile is not None:
                    self.wavfile.close()
                    
                    # Check if the previous conversion process has completed
                    if self.sox_process is not None:
                        self.sox_process.poll()
                        if self.sox_process.returncode is None:
                            print("WARNING: command has not terminated yet:",
                                   cmd, file=sys.stderr)
                        elif self.sox_process.returncode == 0:
                            print("Previous conversion successfully completed.")
                        else:
                            print("WARNING: Previous conversion FAILED.",
                                   cmd, file=sys.stderr)
                            
                    base_filename = wavfile_name.rpartition('.')[0]
                    
                    # Run new sox process to compress wav file                
                    # sox is a high quality audio conversion program
                    # the -v -L rate option forces very high quality
                    # with linear phase.
                    cmd = ("sox --no-dither {filename}.wav --bits 24"
                           " --compression 8 {filename}.flac"
                           " rate -v -L {downsampled_rate}"
                           " && rm {filename}.wav"
                           .format(filename=base_filename,
                                   downsampled_rate=config.DOWNSAMPLED_RATE))
    
                    print("Running", cmd)                                
                    self.sox_process = subprocess.Popen(cmd, shell=True)
                    
                wavfile_name = scpm.get_wavfile_name(adc_data.time)
                self.wavfile = scpm.get_wavfile(wavfile_name)
                
            prev = t
            self.wavfile.writeframes(adc_data.data) # Dump high res. ADC data to disk

    def terminate(self):
        print("\nRecorder shutdown.")
        
        if self.wavfile is not None:
            self.wavfile.close()
            
        if self.sampler is not None:
            self.sampler.terminate()
        
        if self.sox_process is not None:
            self.sox_process.poll()
            if self.sox_process.returncode is None: # sox_process has not terminated yet
                self.sox_process.terminate()


def main():
    recorder = Recorder()

    try:
        recorder.start()
    except KeyboardInterrupt:
        pass
    
    recorder.terminate()


if __name__ == '__main__':
    main()
