#! /usr/bin/python

"""
Compare calculated data with data from a Watts Up.
"""

from __future__ import print_function, division
import time
import snd_card_power_meter.scpm as scpm
from snd_card_power_meter.sampler import Sampler
from snd_card_power_meter.wattsup import WattsUp
import snd_card_power_meter.config as config
import logging
log = logging.getLogger("scpm")

def main():
    scpm.init_logger()
    sampler = Sampler()
    wu = WattsUp()
    
    try:
        sampler.open()
        sampler.start()
        wu.open()        
        calibration = scpm.load_calibration_file()
        log.info("Comparing SCPM data with WattsUp data..."
                 " press CTRL+C to stop.")
        while True:
            wu_data = wu.get() # blocking
            if wu_data is None:
                log.warn("wu_data is none!")
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
            else:
                log.warn("Could not find ADC data for time {}"
                         .format(wu_data.time))

    except KeyboardInterrupt:
        pass
    except Exception:
        log.exception("")
        wu.terminate()
        sampler.terminate()
        raise
    
    wu.terminate()
    sampler.terminate()

if __name__ == '__main__':
    main()
