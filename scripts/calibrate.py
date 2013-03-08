#! /usr/bin/python

"""
Calibration with a Watts Up
---------------------------

NEW CALIBRATION TECHNIQUE:
    Plug in a WattsUp
        run ./calibrate.py
    This will update (or create) a config.CALIBRATION_FILENAME file containing 
    the calibration parameters.

OLD CALIBRATION TECHNIQUE:

Log data from Watts Up using
   screen -dmL wattsup ttyUSB0 volts
"""

from __future__ import print_function, division
import snd_card_power_meter.scpm as scpm
import snd_card_power_meter.sampler as sampler

def main():
    adc_data_queue, audio_stream = sampler.start()
    try:
        scpm.calibrate(adc_data_queue)
    except KeyboardInterrupt:
        pass
    finally:
        audio_stream.stop_stream()
        audio_stream.close()

if __name__ == '__main__':
    main()
