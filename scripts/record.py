#! /usr/bin/python

from __future__ import print_function, division
import datetime, subprocess
import snd_card_power_meter.scpm as scpm
import snd_card_power_meter.sampler as sampler
import snd_card_power_meter.config as config

def main():
    try:
        calibration = scpm.load_calibration_file()
        adc_data_queue, audio_stream = sampler.start()
        filter_state = None
        wavfile = None
        proc = None # a subprocess.Popen object
        cmd = ""        
        while True:
            adc_data = adc_data_queue.get()
            split_adc_data = scpm.split_channels(adc_data)
            power = scpm.calculate_power(split_adc_data)
            # TODO: do save power to disk
            
            # WAV dump to disk
            # Check if it's time to create a new data file
            t = datetime.datetime.fromtimestamp(adc_data.time)
            if wavfile is None or t.minute == (prev.minute+1)%60:
                # TODO: change to hourly recordings after testing!
                if wavfile is not None:
                    wavfile.close()
                    
                    # Check if the previous conversion process has completed
                    if proc is not None:
                        proc.poll()
                        if proc.returncode is None:
                            print("WARNING: command has not terminated yet:",
                                   cmd, file=sys.stderr)
                        elif proc.returncode == 0:
                            print("Previous conversion successfully completed.")
                        else:
                            print("WARNING: Previous conversion FAILED.",
                                   cmd, file=sys.stderr)
                            
                    # Run new shell process to compress wav file
                    base_filename = wavfile_name.rpartition('.')[0]
                    # sox is a high quality audio conversion program
                    # the -v -L rate option forced very high quality
                    # with linear phase.
                    cmd = ("sox --no-dither {filename}.wav --bits 24"
                           " --compression 8 {filename}.flac"
                           " rate -v -L {downsampled_rate}"
                           " && rm {filename}.wav"
                           .format(filename=base_filename,
                                   downsampled_rate=config.DOWNSAMPLED_RATE))

                    print("Running", cmd)                                
                    proc = subprocess.Popen(cmd, shell=True)
                    
                wavfile_name = scpm.get_wavfile_name(adc_data.time)
                wavfile = scpm.get_wavfile(wavfile_name)
                
            prev = t
            
            wavfile.writeframes(adc_data.data)
            
    except KeyboardInterrupt:
        pass
    finally:
        wavfile.close()
        audio_stream.stop_stream()
        audio_stream.close()
        if proc is not None:
            proc.poll()
            if proc.returncode is None: # proc has not terminated yet
                proc.terminate()

if __name__ == '__main__':
    main()
