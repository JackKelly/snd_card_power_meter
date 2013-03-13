""" 

Requirements
------------

  Ubuntu packages
     sudo apt-get install python-dev python-pip alsa alsa-tools libasound2-dev
     libportaudio2 portaudio19-dev sox

  pyaudio

  sox  http://sox.sourceforge.net

  On Linux, add yourself to the audio users:
     sudo adduser USERNAME audio
     (then log out and log back in for this change to take effect)
    
"""

from __future__ import print_function, division
import numpy as np
import wave
import subprocess
import sys
import audioop # docs: http://docs.python.org/2/library/audioop.html
import time
import wattsup
import collections
import ConfigParser # docs: http://docs.python.org/2/library/configparser.html
import datetime
try:
    from Queue import Queue, Empty
except ImportError:
    from queue import Queue, Empty # python 3.x

import sampler, config
from bunch import Bunch

# Named tuples
TVI = collections.namedtuple('TVI', ['time', 'voltage', 'current'])
Power = collections.namedtuple('Power', ['time', 'real', 'apparent', 'v_rms'])

        
def split_channels(stereo):
    """
    Args:
        stereo (binary string): stereo raw ADC data
    
    Returns:
        A Bunch with fields:
        - voltage (binary string): raw ADC data 
        - current (binary string): raw ADC data
    """
    data = Bunch()
    data.voltage = audioop.tomono(stereo, config.SAMPLE_WIDTH, 1, 0)
    data.current = audioop.tomono(stereo, config.SAMPLE_WIDTH, 0, 1)
    return data


def convert_adc_to_numpy_float(split_adc_data):
    """Convert from binary string to numpy array.
    
    Args:
        adc_data: a Bunch with fields:
        - voltage (binary string): raw voltage data from ADC
        - current (binary string): raw current data from ADC
        
    Returns:
        voltage, current (each a float64 numpy vector)
    """
    d = 'int{:d}'.format(config.SAMPLE_WIDTH * 8) # dtype
    voltage = np.fromstring(split_adc_data.voltage, dtype=d).astype(np.float64)
    current = np.fromstring(split_adc_data.current, dtype=d).astype(np.float64)
    
    return voltage, current    


def shift_phase(voltage, current, calibration):
    """Shift voltage and current by PHASE_DIFF number of samples.
    The aim is to correct for phase errors caused by the measurement system.
    
    Args:
        voltage, current (numpy arrays)
        calibration: Bunch with fields:
            - phase_diff
    """    
    pd = abs(int(round(calibration.phase_diff)))
    if pd == 0:
        pass # do nothing if the phase difference is zero
    elif calibration.phase_diff < 0: # current leads voltage
        voltage = voltage[pd:]
        current = current[:-pd]
    else:
        voltage = voltage[:-pd]
        current = current[pd:]
        
    return voltage, current


def calculate_power(split_adc_data, calibration):
    """
    Args:
        split_adc_data: Bunch with fields:
        - voltage (binary string): raw ADC data
        - current (binary string): raw ADC data
        
        calibration: Bunch with fields:
        - watts_per_adc_step (float)
        - volts_per_adc_step (float)
        - amps_per_adc_step (float)
        
    Returns:
        Bunch with fields:
        - real_power (float)
        - apparent_power (float)
        - v_rms (float)
    """
    voltage, current = convert_adc_to_numpy_float(split_adc_data)
    voltage, current = shift_phase(voltage, current, calibration)
    
    data = Bunch()
    
    inst_power = voltage * current # instantaneous power
    data.real_power = inst_power.mean() * calibration.watts_per_adc_step
    if data.real_power < 0:
        data.real_power = 0
    
    data.v_rms = audioop.rms(split_adc_data.voltage, config.SAMPLE_WIDTH)
    data.v_rms *= calibration.volts_per_adc_step
    i_rms = audioop.rms(split_adc_data.current, config.SAMPLE_WIDTH)
    i_rms *= calibration.amps_per_adc_step
    data.apparent_power = data.v_rms * i_rms
    
    power_factor = data.real_power / data.apparent_power
    
    # TODO: leading / lagging phase
    print("real = {:4.2f}W, apparent = {:4.2f}VA, "
          "PF = {:1.3f}, v_rms = {:4.2f}V, i_rms = {:4.4f}A"
          .format(data.real_power, data.apparent_power, power_factor,
                  data.v_rms * calibration.volts_per_adc_step, 
                  i_rms * calibration.amps_per_adc_step))
    print("raw v_rms =", data.v_rms, ", raw i_rms = ", i_rms)
    
    return data
  

def plot(voltage, current, calibration=None):
    """
    Args:
        voltage (numpy array containing raw ADC values)
        
        current (numpy array containing raw ADC values)
        
        calibration: Optional. Bunch with fields:
        - amps_per_adc_step (float)
        - volts_per_adc_step (float)
    """
    import matplotlib.pyplot as plt
    
    if calibration:
        print("volts_per_adc_step =", calibration.volts_per_adc_step)
        print("amps_per_adc_step =", calibration.amps_per_adc_step)
        v_unit = "V"
        i_unit = "A"        
        voltage *= calibration.volts_per_adc_step
        current *= calibration.amps_per_adc_step
    else:
        v_unit = " raw ADC float"
        i_unit = " raw ADC float"
        
    print("VOLTAGE: mean = {:3.3f}{:s}".format(voltage.mean(), v_unit)) 
    print("CURRENT: mean = {:3.3f}{:s}".format(current.mean(), i_unit))

    def center_yaxis(ax):
        NUM_TICKS = 9
        ymin, ymax = ax.get_ylim()
        largest = max(abs(ymin), abs(ymax))
        ax.set_yticks(np.linspace(-largest, largest, NUM_TICKS))

    # two scales code adapted from matplotlib.org/examples/api/two_scales.html
    fig = plt.figure()
    v_ax = fig.add_subplot(111) # v_ax = voltage axes
    v_ax.plot(voltage, "b-")
    center_yaxis(v_ax)    
    v_ax.set_xlabel("time (samples)")
    v_ax.set_title("Voltage and Current waveforms")
    
    # Make the y-axis label and tick labels match the line colour.
    v_ax.set_ylabel("potential different ({:s})".format(v_unit), color="b")
    for tl in v_ax.get_yticklabels():
        tl.set_color("b")
    
    # The function twinx() give us access to a second plot that
    # overlays the graph ax2 and shares the same X axis, but not the Y axis.     
    i_ax = v_ax.twinx()
    i_ax.plot(current, "g-")
    center_yaxis(i_ax)
    i_ax.set_ylabel("current ({:s})".format(i_unit), color="g")
    for tl in i_ax.get_yticklabels():
        tl.set_color("g")

    plt.grid()
    plt.show()
    
    
def find_time(adc_data_queue, target_time):
    """
    Looks through adc_data_queue for an entry with time=t.
    If an entry is found then that entry is returned and that entry and all
    previous entries are deleted from the Queue.
    If no matching entry is found the None is returned.
    
    Beware: If the queue contains at least one entry then EVERY call to this
        function will take at least one item off the queue, even if that
        item is not returned! 
    
    Args:
        adc_data_queue (Queue)
        target_time (int or float): UNIX timestamp
    """
    t = 0
    target_time = int(round(target_time))
    while t < target_time:
        adc_data = None                    
        try:
            adc_data = adc_data_queue.get_nowait()
        except Empty:
            return None
        else:            
            t = int(round(adc_data.time))
    
    if t == target_time:
        return adc_data
    else:
        return None
    
    
def positive_zero_crossings(data):
    """Returns indices of positive-heading zero crossings.
    
    Args:
        data (numpy vector)
    """
    # Adapted from Jim Brissom's SO answer: http://stackoverflow.com/a/3843124
    return np.where(np.diff(np.sign(data)) > 0)[0]
    

class ZeroCrossingError(Exception):
    pass


def get_phase_diff(split_adc_data, tolerance):
    """Finds the phase difference between the positive-going zero crossings
    of the voltage and current waveforms.
    
    Args:
        split_adc_data: Bunch with fields:
            - voltage (binary string): raw adc data
            - current (binary string): raw adc data
        tolerance (float): max number of samples by which 
            i and v zero crossings can differ.
    
    Returns:
        The mean number of samples by which the zero crossings of the current
        and voltage waveforms differ.  Negative means current leads voltage.
    """ 
    
    voltage, current = convert_adc_to_numpy_float(split_adc_data)    
    vzc = positive_zero_crossings(voltage) # vzc = voltage zero crossings
    izc = positive_zero_crossings(current) # izc = current zero crossings
    
    # sanity check length
    if not (len(izc)-2 < len(vzc) < len(izc)+2):
        raise ZeroCrossingError("ERROR: number of current zero crossings ({})"
                                " too dissimilar to\n"
                                "       number of voltage zero crossings ({})."
                                .format(len(izc), len(vzc)))

    
    # go through each zero crossing in turn and compare
    i_offset = 0
    phase_diffs = []
    for i in range(len(vzc)):
        try:
            phase_diff = izc[i+i_offset] - vzc[i]
        except IndexError:
            continue
        
        if abs(phase_diff) < tolerance: # This is a valid comparison            
            phase_diffs.append(phase_diff)
        elif phase_diff > tolerance:
            i_offset -= 1
        else:
            i_offset += 1
            
    print("phase diff = {:.1f} samples, std = {:.3f}".format(
                                                       np.mean(phase_diffs),
                                                       np.std(phase_diffs)))
    
    return np.mean(phase_diffs)


def load_calibration_file():
    """Loads config.CALIBRATION_FILENAME.
    
    Returns a Bunch with fields:
        - volts_per_adc_step (float)
        - amps_per_adc_step (float)
        - phase_diff (float)
        - watts_per_adc_step (float)
    """
    
    config_parser = ConfigParser.RawConfigParser()
    config_parser.read(config.CALIBRATION_FILENAME)
    calib = Bunch()
    calib_section = "Calibration"
    try:
        calib.volts_per_adc_step = config_parser.getfloat(calib_section, 
                                                          "volts_per_adc_step")
        calib.amps_per_adc_step = config_parser.getfloat(calib_section,
                                                         "amps_per_adc_step")
        calib.phase_diff = config_parser.getfloat(calib_section,
                                                  "phase_difference")
    except (ConfigParser.NoOptionError, ConfigParser.NoSectionError) as e:
        print("Error loading option from config file", str(e), file=sys.stderr)
        return None
    else:
        calib.watts_per_adc_step = (calib.volts_per_adc_step * 
                                    calib.amps_per_adc_step)   
        print("VOLTS_PER_ADC_STEP =", calib.volts_per_adc_step)
        print("AMPS_PER_ADC_STEP  =", calib.amps_per_adc_step)
        print("WATTS_PER_ADC_STEP =", calib.watts_per_adc_step)

    return calib


def calibrate(adc_data_queue, wu):
    """
    Args:
        adc_data_queue (Queue)
        wu (WattsUp): must already be opened
    """
    
    v_acumulator = np.float64(0.0) # accumulator for voltage
    i_acumulator = np.float64(0.0) # accumulator for current
    pd_acumulator = np.float64(0.0) # accumulator for phase difference
    n_v_samples = 0 # number of voltage samples
    n_i_samples = 0 # number of current samples
    n_pd_samples = 0 # number of phase diff samples
    av_i_calibration = np.float64(0.0) # average current calibration
    av_v_calibration = np.float64(0.0) # average voltage calibration
    adc_i_rms = 0.0
    av_pd = np.float64(0.0) # average phase diff
    
    config_parser = ConfigParser.RawConfigParser()
    config_parser.read(config.CALIBRATION_FILENAME)
    try:
        config_parser.add_section("Calibration")
    except ConfigParser.DuplicateSectionError:
        print("Overwriting existing calibration file",
              config.CALIBRATION_FILENAME)
    
    while True:
        wu_data = wu.get() # blocking
        
        # Now get ADC data recorded at wu_data.time
        adc_data = find_time(adc_data_queue, wu_data.time)
        
        if adc_data:
            split_adc_data = split_channels(adc_data.data)            
            
            # Voltage
            adc_v_rms = audioop.rms(split_adc_data.voltage, config.SAMPLE_WIDTH) 
            n_v_samples += 1
            v_acumulator += wu_data.volts / adc_v_rms
            av_v_calibration = v_acumulator / n_v_samples
            
            # Current
            adc_i_rms = audioop.rms(split_adc_data.current, config.SAMPLE_WIDTH)            
            if wu_data.amps > 0.1:
                n_i_samples += 1 
                i_acumulator += wu_data.amps / adc_i_rms
                av_i_calibration = i_acumulator / n_i_samples
                
                # Phase difference calc
                if wu_data.power_factor > 0.97:
                    try:
                        pd_acumulator += get_phase_diff(split_adc_data)
                    except ZeroCrossingError as e:
                        print(str(e), file=sys.stderr)
                        processing_pf = False
                    else:
                        processing_pf = True
                        n_pd_samples += 1
                        av_pd = pd_acumulator / n_pd_samples
                        print("Average phase diff =", av_pd)
                        print(av_pd.dtype)
                        config_parser.set("Calibration", "phase_difference",
                                          av_pd)
                else:
                    processing_pf = False

                print("Watts up reports power factor "
                      "is {:2.2f} so".format(wu_data.power_factor),
                      "will" if processing_pf else "will not",
                      "calibrate phase angle...")

            else:
                print("Not sampling amps because the WattsUp reading too low.")    
            
            print("WattsUp:    volts = {:>03.2f}, amps = {:>02.2f}, \n"
                  "Calculated: volts = {:>03.2f}, amps = {:>02.2f}, \n"
                  "Raw ADC: voltage = {}, current = {}, \n"
                  "v_calibration = {}, i_calibration = {}, \n"
                  "adc time = {}, watts up time = {}, time diff = {:1.3f}s \n"
                  .format(wu_data.volts, wu_data.amps,
                          adc_v_rms * av_v_calibration,
                          adc_i_rms * av_i_calibration,
                          adc_v_rms, adc_i_rms,
                          av_v_calibration, av_i_calibration,
                          adc_data.time, wu_data.time,
                          adc_data.time - wu_data.time))

            config_parser.set("Calibration", "volts_per_adc_step",
                              av_v_calibration)
            config_parser.set("Calibration", "amps_per_adc_step",
                              av_i_calibration)
            with open(config.CALIBRATION_FILENAME, "wb") as calibfile:
                config_parser.write(calibfile)
        else:
            print("Could not find match for", wu_data.time, file=sys.stderr)
            time.sleep(config.RECORD_SECONDS * 2)

    
def get_wavfile_name(t):
    return config.FLAC_FILENAME_PREFIX + "-{:f}".format(t) + ".wav" 


def get_wavfile(wavefile_name):
    wavfile = wave.open(wavefile_name, 'wb')
    wavfile.setnchannels(config.N_CHANNELS)
    wavfile.setsampwidth(config.SAMPLE_WIDTH)
    wavfile.setframerate(config.FRAME_RATE)
    return wavfile    

