#! /usr/bin/python

"""
Data written to config.DATA_FILENAME. Multiple columns separated by a space:
 1. UNIX timestamp
 2. real power (watts)
 3. apparent_power (VA)
 4. volts RMS
 5. phase_diff (degrees) +ve means current leads voltage (capacitive)
"""

from __future__ import print_function, division
import datetime, subprocess
import logging
log = logging.getLogger("scpm")
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
        log.info("Recording power data. Press CTRL+C to stop.")
        while True:
            adc_data = self.sampler.adc_data_queue.get()
            if adc_data is None:
                log.warn("adc_data was None!")
                continue
            
            split_adc_data = scpm.split_channels(adc_data.data)
            
            # Calculate real power, apparent power, v_rms and write to disk
            adc_rms = scpm.calculate_adc_rms(split_adc_data)
            power = scpm.calculate_calibrated_power(split_adc_data, adc_rms, 
                                                    calibration)

            with open(config.DATA_FILENAME, 'a') as data_file:
                data_file.write('{:.1f} {:.2f} {:.2f} {:.2f} {:.2f}\n'
                                .format(adc_data.time, power.real_power,
                                        power.apparent_power, power.volts_rms,
                                        power.phase_diff))
    
            # Check if it's time to create a new FLAC file
            t = datetime.datetime.fromtimestamp(adc_data.time)
            if self.wavfile is None or t.hour == (prev.hour+1)%24:
                if self.wavfile is not None:
                    self.wavfile.close()
                    
                    # Check if the previous conversion process has completed
                    if self.sox_process is not None:
                        self.sox_process.poll()
                        if self.sox_process.returncode is None:
                            log.warn("WARNING: command has not terminated yet: "
                                     + cmd)
                        elif self.sox_process.returncode == 0:
                            log.debug("Previous conversion successfully completed: "
                                      + cmd)
                        else:
                            log.warn("WARNING: Previous conversion FAILED: "
                                     + cmd)
                            
                    base_filename = wavfile_name.rpartition('.')[0]
                    
                    # Run new sox process to compress wav file                
                    # sox is a high quality audio conversion program
                    # the "rate -v -L" option forces very high quality
                    # rate conversion with linear phase.
                    cmd = ("sox --no-dither {filename}.wav --bits 24"
                           " --compression 8 {filename}.flac"
                           " rate -v -L {downsampled_rate}"
                           " && rm {filename}.wav"
                           .format(filename=base_filename,
                                   downsampled_rate=config.DOWNSAMPLED_RATE))
    
                    log.info("Running: " + cmd)                                
                    self.sox_process = subprocess.Popen(cmd, shell=True)
                    
                wavfile_name = scpm.get_wavfile_name(adc_data.time)
                self.wavfile = scpm.get_wavfile(wavfile_name)
                
            prev = t
            self.wavfile.writeframes(adc_data.data) # Dump high res. ADC data to disk

    def terminate(self):
        log.info("Recorder shutdown.\n")
        
        if self.wavfile is not None:
            self.wavfile.close()
            
        if self.sampler is not None:
            self.sampler.terminate()
        
        if self.sox_process is not None:
            self.sox_process.poll()
            if self.sox_process.returncode is None: # sox_process has not terminated yet
                self.sox_process.terminate()
                
        logging.shutdown()


def main():
    scpm.init_logger()
    recorder = Recorder()

    try:
        recorder.start()
    except KeyboardInterrupt:
        recorder.terminate()
    except:
        recorder.terminate()
        print("Unexpected exception!")
        raise


if __name__ == '__main__':
    main()
