from __future__ import print_function
import snd_card_power_meter.scpm as scpm
import unittest
import collections
try:
    from Queue import Queue, Empty
except ImportError:
    from queue import Queue, Empty # python 3.x

class TestLoadConfig(unittest.TestCase):

    def setUp(self):
        # self.adc_data_queue, self.adc_thread = scpm.start_adc_data_queue_and_thread()
        pass
    
    def test_find_time(self):
        q = Queue()
        
        va = scpm.find_time(q, 1002)
        self.assertEqual(va, None)
        
        for i in range(1,10):
            q.put(scpm.VA(1000 + i, i, i*10))
        
        va = scpm.find_time(q, 1003)
        self.assertEqual(va, scpm.VA(1003, 3, 30))
           
        va = scpm.find_time(q, 1002)
        self.assertEqual(va, None)
        
        va = scpm.find_time(q, 1003)
        self.assertEqual(va, None)
        
        va = scpm.find_time(q, 1004)
        self.assertEqual(va, None)

        va = scpm.find_time(q, 1009)
        self.assertEqual(va, scpm.VA(1009, 9, 90))
        
        va = scpm.find_time(q, 9999)
        self.assertEqual(va, None)


if __name__ == '__main__':
    unittest.main()
