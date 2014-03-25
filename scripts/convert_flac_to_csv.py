#! /usr/bin/python
"""
Plot power data from a FLAC or WAV file.

Dependencies
------------

sox (to uncompress the FLAC files and convert to 32-bit WAVs)

"""

from __future__ import print_function, division
import snd_card_power_meter.scpm as scpm
import snd_card_power_meter.config as config
from snd_card_power_meter.bunch import Bunch 
import argparse, logging, subprocess, wave, os, sys
from plot_flac import uncompress_and_load, get_adc_data
log = logging.getLogger("scpm")

def setup_argparser():
    # Process command line _args
    parser = argparse.ArgumentParser(description="Load and plot power data "
                                                 "from FLAC or WAV file.")
       
    parser.add_argument('input_file', help='FLAC or WAV file to read. NO suffix!')

    parser.add_argument('--calibration-file', dest='calibration_file',
                        default=config.CALIBRATION_FILENAME)
    
    parser.add_argument('--correct-phase', action='store_true')

    args = parser.parse_args()
    return args

def convert_flac_file(input_filename, output_filename, args):
    wavfile = uncompress_and_load(input_filename)
    samples_per_degree = (wavfile.getframerate() / config.MAINS_HZ) / 360
    calibration = scpm.load_calibration_file(filename=args.calibration_file,
                                             samples_per_degree=samples_per_degree)
    if calibration is None:
        raise Exception("No calibration data loaded.")

    adc_data = get_adc_data(wavfile)
    time = os.path.split(args.input_file)[1]
    time = time.replace('vi-','')
    time = time.replace('_', '.')
    time = float(time)
    while adc_data.data:
        adc_data.sample_width = wavfile.getsampwidth()
        adc_data = scpm.join_frames_and_widen(adc_data)
        split_adc_data = scpm.split_channels(adc_data)
        adc_rms = scpm.calculate_adc_rms(split_adc_data)
        voltage, current = scpm.convert_adc_to_numpy_float(split_adc_data)
        if args.correct_phase:
            voltage, current = scpm.shift_phase(voltage, current, calibration)
        calcd_data = scpm.calculate_calibrated_power(split_adc_data, 
                                                adc_rms, calibration)

        # Dump power data to disk
        with open(output_filename, 'a') as data_file:
            data_file.write('{:.1f} {:.2f} {:.2f} {:.2f}\n'
                            .format(time, calcd_data.real_power,
                                    calcd_data.apparent_power, 
                                    calcd_data.volts_rms))

        adc_data = get_adc_data(wavfile)
        time += len(voltage) / wavfile.getframerate()

    wavfile.close()

def main():
    args = setup_argparser()
    scpm.init_logger()
    log.setLevel(logging.ERROR)

    # TODO: 
    # delete output file
    # find all input files
    # loop through them
    convert_flac_file(args.input_file, args.input_file + '.dat', args)


if __name__ == '__main__':
    main()
