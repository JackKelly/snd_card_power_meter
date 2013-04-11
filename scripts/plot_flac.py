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
log = logging.getLogger("scpm")

def uncompress_and_load(flac_filename):
    """
    Args:
        flac_filename (str): filename of FLAC file.  NO suffix.
        
    Returns:
        Wave_read object
    """

    if not os.path.exists(flac_filename + '.wav'):
        print("Uncompressing", flac_filename)
        # Without '-t wavpcm' wave.open complains of a format error; because
        # without '-t wavpcm', sox creates an MS Extensible WAV, not a
        # standard PCM WAV.  See here: 
        # http://code.google.com/p/opencinematools/issues/detail?id=29
        cmd = ('sox {filename}.flac --bits 32 -t wavpcm {filename}.wav'
               ' && soxi -V9 {filename}.wav'
               .format(filename=flac_filename))
        print("Running", cmd)
        flac_process = subprocess.Popen(cmd, shell=True)
        flac_process.wait()
        if flac_process.returncode != 0:
            print("decompression error!")
            sys.exit(1)
        print("done running decompression process.")

    wavfile = wave.open(flac_filename + '.wav', 'r')
    print("Opened {}.wav".format(flac_filename))
    d = {}
    d['nchannels'], d['sampwidth'], d['framerate'], \
    d['nframes'], d['comptype'], d['compname'] = wavfile.getparams()
    print("nchannels={nchannels}, sampwidth={sampwidth},"
          " framerate={framerate}Hz, nframes={nframes},\n"
          "comptype={comptype}, compname={compname}"
          .format(**d))
    return wavfile


def get_adc_data(wavfile):
    n_reads_per_queue_item = int(round((wavfile.getframerate() / 
                                       config.FRAMES_PER_BUFFER) * 
                                       config.RECORD_SECONDS))
    frames = []
    for _ in range(n_reads_per_queue_item):
        frames.append(wavfile.readframes(config.FRAMES_PER_BUFFER))
        
    return Bunch(data=b''.join(frames))


def setup_argparser():
    # Process command line _args
    parser = argparse.ArgumentParser(description="Load and plot power data "
                                                 "from FLAC or WAV file.")
       
    parser.add_argument('input_file', help='FLAC or WAV file to read. NO suffix!')

    parser.add_argument('--calibration-file', dest='calibration_file')
    
    parser.add_argument('--correct-phase', action='store_true')

    args = parser.parse_args()
    return args

def main():
    scpm.init_logger()
    
    args = setup_argparser()
    
    wavfile = uncompress_and_load(args.input_file)
    sample_width = wavfile.getsampwidth()
    samples_per_degree = (wavfile.getframerate() / config.MAINS_HZ) / 360
    
    calibration = scpm.load_calibration_file(filename=args.calibration_file,
                                             samples_per_degree=samples_per_degree)
        
    adc_data = get_adc_data(wavfile)
    split_adc_data = scpm.split_channels(adc_data.data, sample_width)
    adc_rms = scpm.calculate_adc_rms(split_adc_data, sample_width)
    calcd_data = scpm.calculate_calibrated_power(split_adc_data, 
                                            adc_rms, calibration)
    
    print("")
    scpm.print_power(calcd_data)
    voltage, current = scpm.convert_adc_to_numpy_float(split_adc_data)
    if args.correct_phase:
        print("Correcting phase by {} degrees = {} samples."
              .format(calibration.phase_diff, calibration.phase_diff_n_samples))
        voltage, current = scpm.shift_phase(voltage, current, calibration)
    else:
        print("Not correcting phase difference"
              " (use --correct-phase to correct phase)")

    scpm.plot(voltage, current, calibration)

    wavfile.close()

if __name__ == '__main__':
    main()
