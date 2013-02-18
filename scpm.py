#! /usr/bin/python

""" 
THIS CODE IS JUST ME LEARNING TO USE ALSAAUDIO.
IT DOESN'T YET DO ANY POWER MEASUREMENT STUFF! 
"""

# TODO:
#  * see if wave.writeframes() does proper down-sampling
#  * do power calcs

from __future__ import print_function, division
import numpy as np
import alsaaudio # docs: http://pyalsaaudio.sourceforge.net/libalsaaudio.html
import wave
import matplotlib.pyplot as plt

inp = alsaaudio.PCM(alsaaudio.PCM_CAPTURE)
inp.setchannels(1)
inp.setrate(44100) # Hz
inp.setformat(alsaaudio.PCM_FORMAT_S32_LE) # signed 32-bit Little Endian
inp.setperiodsize(1024) # in frames

w = wave.open('test.wav', 'w')
w.setnchannels(1)
w.setsampwidth(2)
w.setframerate(44100)

#while True:
length, data = inp.read()
a = np.fromstring(data, dtype='int32')
#print("{:.1f}".format(np.abs(a).mean()))
plt.plot(a)
plt.show()
w.writeframes(data)
