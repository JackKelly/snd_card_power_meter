#! /usr/bin/python
"""
Plot data from sound card.  Does not do any phase correction, neither
for the plotted data nor for the printed data.
"""

from __future__ import print_function, division
import snd_card_power_meter.scpm as scpm
import argparse, logging, subprocess, shlex, wave
log = logging.getLogger("scpm")

def uncompress_and_load(flac_filename):
    """
    Args:
        flac_filename (str): filename of FLAC file.  NO suffix.
        
    Returns:
        Wave_read object
    """
    cmd = 'sox {filename}.flac {filename}.wav'.format(filename=flac_filename)
    print("Running", cmd)
    sox_process = subprocess.Popen(shlex.split(cmd))
    sox_process.wait()
    wavfile = wave.open(flac_filename + '.wav', 'rb')
    print("Opened {}.wav".format(flac_filename))
    return wavfile


def setup_argparser():
    # Process command line _args
    parser = argparse.ArgumentParser(description="Load and plot power data "
                                                 "from FLAC file.")
       
    parser.add_argument('input_file', help='FLAC file to read. NO suffix!')

    args = parser.parse_args()
    return args

def main():
    scpm.init_logger()
    
    args = setup_argparser()
    
    wavfile = uncompress_and_load(args.input_file)
    
#    try:
#        sampler.open()
#        sampler.start()
#        calibration = scpm.load_calibration_file()                
#        adc_data = sampler.adc_data_queue.get()
#    except KeyboardInterrupt:
#        sampler.terminate()
#    else:
#        sampler.terminate()
#        split_adc_data = scpm.split_channels(adc_data.data)
#        adc_rms = scpm.calculate_adc_rms(split_adc_data)
#        calcd_data = scpm.calculate_calibrated_power(split_adc_data, 
#                                                adc_rms, calibration)
#        
#        print("")
#        scpm.print_power(calcd_data)
#        voltage, current = scpm.convert_adc_to_numpy_float(split_adc_data)
#        scpm.plot(voltage, current, calibration)
#        logging.shutdown()

    wavfile.close()

if __name__ == '__main__':
    main()
