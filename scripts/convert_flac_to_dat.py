#! /usr/bin/python
from __future__ import print_function, division
import snd_card_power_meter.scpm as scpm
import snd_card_power_meter.config as config
from snd_card_power_meter.bunch import Bunch 
import argparse, logging, subprocess, wave, os, sys
from plot_flac import uncompress_and_load, get_adc_data
log = logging.getLogger("scpm")

DESCRIPTION = """
Takes a whole directory of FLAC/WAV files and converts them
all to a single .DAT file.  Or can take a single file.

Requires 'sox' (to uncompress the FLAC files and convert to 32-bit WAVs).
"""

def convert_flac_file(input_filename, output_filename, calibration_filename, correct_phase):
    wavfile = uncompress_and_load(input_filename)
    samples_per_degree = (wavfile.getframerate() / config.MAINS_HZ) / 360
    calibration = scpm.load_calibration_file(filename=calibration_filename,
                                             samples_per_degree=samples_per_degree)
    if calibration is None:
        raise Exception("No calibration data loaded.")

    adc_data = get_adc_data(wavfile)
    time = os.path.split(input_filename)[1]
    time = time.replace('vi-','')
    time = time.replace('_', '.')
    time = float(time)
    while adc_data.data:
        adc_data.sample_width = wavfile.getsampwidth()
        adc_data = scpm.join_frames_and_widen(adc_data)
        split_adc_data = scpm.split_channels(adc_data)
        adc_rms = scpm.calculate_adc_rms(split_adc_data)
        voltage, current = scpm.convert_adc_to_numpy_float(split_adc_data)
        if correct_phase:
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


def setup_argparser():
    # Process command line _args
    parser = argparse.ArgumentParser(description=DESCRIPTION)
       
    parser.add_argument('input', help='FLAC or WAV file (NO suffix) or directory of FLACs.')

    parser.add_argument('output', help='Output DAT file (with suffix).')

    parser.add_argument('calibration_file',
                        help='The calibration.cfg file.')
    
    parser.add_argument('--correct-phase', action='store_true')

    parser.add_argument('--do-not-delete-output', action='store_true')

    args = parser.parse_args()
    return args


def get_flac_filenames(directory):
    all_filenames = os.listdir(directory)
    flac_filenames = [os.path.join(directory, os.path.splitext(fname)[0])
                      for fname in all_filenames if fname.endswith('.flac')]
    return flac_filenames


def main():
    args = setup_argparser()

    if not os.path.exists(args.input):
        raise Exception("Input '{}' does not exist".format(args.input))

    scpm.init_logger()
    log.setLevel(logging.ERROR)

    # Process args.input
    if os.path.isdir(args.input):
        filenames = get_flac_filenames(args.input)
        filenames.sort()
        print(filenames)
    else:
        filenames = [args.input]

    # Delete output fname if it exists
    if not args.do_not_delete_output and os.path.exists(args.output):
        log.info("Deleting '{}'".format(args.output))
        os.remove(args.output)

    for fname in filenames:
        convert_flac_file(fname, args.output, args.calibration_file, args.correct_phase)


if __name__ == '__main__':
    main()
