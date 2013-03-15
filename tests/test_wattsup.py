from __future__ import print_function, division
import snd_card_power_meter.wattsup as wattsup 
import unittest

class TestLoadConfig(unittest.TestCase):

    def setUp(self):
        pass
    
    def test_parse_wu_line(self):
        line = "[14:51:42] 50.7, 227.4, 39.2, 5.5"
        data = wattsup._parse_wu_line(line)
        self.assertEqual(data.real_power, 50.7)
        self.assertEqual(data.volts_rms, 227.4)
        self.assertEqual(data.amps_rms, 0.392)
        self.assertEqual(data.power_factor, 0.55)
        self.assertEqual(data.apparent_power, 50.7 / 0.55)
        
        line = "[14:51:42] 50.7, 227.4, 39.2, 5.5baddness"
        data = wattsup._parse_wu_line(line)
        self.assertEqual(data.real_power, 50.7)
        self.assertEqual(data.volts_rms, 227.4)
        self.assertEqual(data.amps_rms, 0.392)
        self.assertEqual(data.power_factor, 0.55)
        self.assertEqual(data.apparent_power, 50.7 / 0.55)
        
        line = "[14:51:42] 50.7, 227.4, 39.2, 5.5baddness: isn't this bad"
        data = wattsup._parse_wu_line(line)
        self.assertEqual(data.real_power, 50.7)
        self.assertEqual(data.volts_rms, 227.4)
        self.assertEqual(data.amps_rms, 0.392)
        self.assertEqual(data.power_factor, 0.55)
        self.assertEqual(data.apparent_power, 50.7 / 0.55)           

if __name__ == '__main__':
    unittest.main()
