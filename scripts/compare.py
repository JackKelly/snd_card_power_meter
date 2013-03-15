#! /usr/bin/python

"""
Compare calculated data with data from a Watts Up.
"""

from __future__ import print_function, division
import sys, time
import snd_card_power_meter.scpm as scpm
from snd_card_power_meter.sampler import Sampler
from snd_card_power_meter.wattsup import WattsUp
import snd_card_power_meter.config as config

def main():
    sampler = Sampler()
    wu = WattsUp()
    
    try:
        sampler.open()
        sampler.start()
        wu.open()        
        calibration = scpm.load_calibration_file()
        
        while True:
            wu_data = wu.get() # blocking
            if wu_data is None:
                continue
            
            time.sleep(config.RECORD_SECONDS)
            # Now get ADC data recorded at wu_data.time
            adc_data = scpm.find_time(sampler.adc_data_queue, wu_data.time)
            if adc_data:
                print("")                
                split_adc_data = scpm.split_channels(adc_data.data)
                adc_rms = scpm.calculate_adc_rms(split_adc_data)
                calcd_data = scpm.calculate_calibrated_power(split_adc_data, 
                                                        adc_rms, calibration)
                
                scpm.print_power(calcd_data, wu_data)

    except KeyboardInterrupt:
        pass
    except Exception:
        print("Exception!  Terminating.", file=sys.stderr)
        wu.terminate()
        sampler.terminate()
        raise
    
    wu.terminate()
    sampler.terminate()

if __name__ == '__main__':
    main()
