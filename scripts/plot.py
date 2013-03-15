#! /usr/bin/python
"""
Plot data from sound card.  Does not do any phase correction, neither
for the plotted data nor for the printed data.
"""

from __future__ import print_function, division
import snd_card_power_meter.scpm as scpm
from snd_card_power_meter.sampler import Sampler

def main():
    sampler = Sampler()
    try:
        sampler.open()
        sampler.start()
        calibration = scpm.load_calibration_file()                
        adc_data = sampler.adc_data_queue.get()
    except KeyboardInterrupt:
        sampler.terminate()
    else:
        sampler.terminate()
        split_adc_data = scpm.split_channels(adc_data.data)
        adc_rms = scpm.calculate_adc_rms(split_adc_data)
        calcd_data = scpm.calculate_calibrated_power(split_adc_data, 
                                                adc_rms, calibration)
        
        print("")
        scpm.print_power(calcd_data)
        voltage, current = scpm.convert_adc_to_numpy_float(split_adc_data)
        scpm.plot(voltage, current, calibration)

if __name__ == '__main__':
    main()
