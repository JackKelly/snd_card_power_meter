from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import snd_card_power_meter.scpm as scpm
import snd_card_power_meter.config as config
from snd_card_power_meter.bunch import Bunch
import unittest

try:
    from Queue import Queue, Empty
except ImportError:
    from queue import Queue, Empty # python 3.x

class TestLoadConfig(unittest.TestCase):

    def setUp(self):
        # self.adc_data_queue, self.adc_thread = scpm.start_adc_data_queue_and_thread()
        pass
    
#    def test_find_time(self):
#        q = Queue()
#        
#        va = scpm.find_time(q, 1002)
#        self.assertEqual(va, None)
#        
#        for i in range(1,10):
#            q.put(scpm.VA(1000 + i, i, i*10))
#        
#        va = scpm.find_time(q, 1003)
#        self.assertEqual(va, scpm.VA(1003, 3, 30))
#           
#        va = scpm.find_time(q, 1002)
#        self.assertEqual(va, None)
#        
#        va = scpm.find_time(q, 1003)
#        self.assertEqual(va, None)
#        
#        va = scpm.find_time(q, 1004)
#        self.assertEqual(va, None)
#
#        va = scpm.find_time(q, 1009)
#        self.assertEqual(va, scpm.VA(1009, 9, 90))
#        
#        va = scpm.find_time(q, 9999)
#        self.assertEqual(va, None)
        

    def test_indicies_of_positive_peaks(self):
        FREQ = 50
        SAMPLES_PER_SECOND = config.FRAME_RATE
        x = np.linspace(0, np.pi*2*FREQ, SAMPLES_PER_SECOND)
        y = np.sin(x)
        calcd_indices = scpm.indices_of_positive_peaks(y, FREQ)
        
        samples_per_cycle = SAMPLES_PER_SECOND / FREQ
        start = int(round(samples_per_cycle / 4))
        stop = int(round(SAMPLES_PER_SECOND - (samples_per_cycle * 3 / 4)))
        correct_indices = np.linspace(start, stop-1, FREQ).round()
        
        print(calcd_indices)
        print(correct_indices)
        
#        plt.plot(y)
#        plt.plot(calcd_indices, np.ones(len(calcd_indices)), 'r*')
#        plt.show()
        
        self.assertTrue((calcd_indices==correct_indices).all())

if __name__ == '__main__':
    unittest.main()
