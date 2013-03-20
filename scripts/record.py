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
        self.wavfile_name = None
        self.sampler = Sampler()

    def start(self):
        calibration = scpm.load_calibration_file()
        self.sampler.open()
        self.sampler.start()
        log.info("Recording power data. Press CTRL+C to stop.")

        prev_conv_time = None # the last time sox was run
        while True:
            adc_data = self.sampler.adc_data_queue.get()
            if adc_data is None:
                log.warn("adc_data was None!")
                continue
            
            # Calculate real power, apparent power, v_rms
            split_adc_data = scpm.split_channels(adc_data.data)
            adc_rms = scpm.calculate_adc_rms(split_adc_data)
            power = scpm.calculate_calibrated_power(split_adc_data, adc_rms, 
                                                    calibration)

            # Dump power data to disk
            with open(config.DATA_FILENAME, 'a') as data_file:
                data_file.write('{:.1f} {:.2f} {:.2f} {:.2f}'
                                .format(adc_data.time, power.real_power,
                                        power.apparent_power, power.volts_rms))
                if power.phase_diff is None:
                    data_file.write('\n')
                else:
                    data_file.write(' {:.2f}\n'.format(power.phase_diff))
    
            # Check if it's time to create a new FLAC file
            t = datetime.datetime.fromtimestamp(adc_data.time)
            if self.wavfile is None or t.hour == (prev_conv_time.hour+1)%24:
                self._close_and_compress_wavfile()
                self.wavfile_name = scpm.get_wavfile_name(adc_data.time)
                self.wavfile = scpm.get_wavfile(self.wavfile_name)
            prev_conv_time = t
            
            # Write uncompressed wav data to disk
            self.wavfile.writeframes(adc_data.data)
            
    def _close_and_compress_wavfile(self):
        if self.wavfile is None:
            return

        self.wavfile.close()
        
        # Check if the previous conversion process has completed
        self._check_sox_process_has_completed()
        
        # Run new sox process to compress wav file                
        # sox is a high quality audio conversion program
        # the "rate -v -L" option forces very high quality
        # rate conversion with linear phase.
        cmd = ("sox --no-dither {filename}.wav --bits 24"
               " --compression 8 {filename}.flac"
               " rate -v -L {downsampled_rate}"
               " && rm {filename}.wav"
               .format(filename=self.wavfile_name,
                       downsampled_rate=config.DOWNSAMPLED_RATE))

        log.info("Running: " + cmd)                                
        self.sox_process = subprocess.Popen(cmd, shell=True,
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE)

    def _check_sox_process_has_completed(self):
        if self.sox_process is None:
            return
        
        self.sox_process.poll()
        if self.sox_process.returncode is None:
            log.warn("Previous sox conversion has not terminated yet!"
                     " Waiting...")
            self.sox_process.wait()

        if self.sox_process.returncode == 0:
            log.info("Previous conversion successfully completed!")
        else:
            log.warn("Previous sox conversion FAILED!")
    
        # Log sox output
        stdout = self.sox_process.stdout.read()
        stderr = self.sox_process.stderr.read()
        if stdout:
            log.debug("sox stdout: {}".format(stdout))
        if stderr:
            log.debug("sox stderr: {}".format(stderr))
        

    def terminate(self):
        log.info("Recorder shutdown.\n")
        
        self._close_and_compress_wavfile()
            
        if self.sampler is not None:
            self.sampler.terminate()
        
        self._check_sox_process_has_completed()
                
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
