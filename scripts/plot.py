#! /usr/bin/python

from __future__ import print_function, division
import snd_card_power_meter.scpm as scpm
import snd_card_power_meter.sampler as sampler

def main():
    calibration = scpm.load_calibration_file()
    adc_data_queue, audio_stream = sampler.start()                        

    adc_data = adc_data_queue.get()
    split_adc_data = scpm.split_channels(adc_data.data)
    voltage, current = scpm.convert_adc_to_numpy_float(split_adc_data)
    scpm.plot(voltage, current, calibration)

    audio_stream.stop_stream()
    audio_stream.close()

if __name__ == '__main__':
    main()
