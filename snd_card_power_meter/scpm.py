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
import sys
import audioop # docs: http://docs.python.org/2/library/audioop.html
import time
import collections
import ConfigParser # docs: http://docs.python.org/2/library/configparser.html
try:
    import Queue as queue
except ImportError:
    import queue # python 3.x

import config
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


def shift_phase(voltage, current, calibration=None):
    """Shift voltage and current by PHASE_DIFF number of samples.
    The aim is to correct for phase errors caused by the measurement system.
    
    Args:
        voltage, current (numpy arrays)
        calibration: Bunch with fields:
            - phase_diff
    """
    if calibration is None or calibration.__dict__.get("phase_diff") is None:
        return voltage, current
    
    pd = abs(int(round(calibration.phase_diff)))
    if pd == 0:
        pass
    elif calibration.phase_diff > 0: # I leads V
        voltage = voltage[pd:]
        current = current[:-pd]
    else:
        voltage = voltage[:-pd]
        current = current[pd:]
        
    return voltage, current


def calculate_adc_rms(split_adc_data):
    """
    Args:
        split_adc_data: Bunch with fields:
        - voltage (binary string): raw ADC data
        - current (binary string): raw ADC data
    
    Returns:
        Bunch with fields:
        - adc_v_rms (float)
        - adc_i_rms (float)    
    """
    data = Bunch()
    data.adc_v_rms = audioop.rms(split_adc_data.voltage, config.SAMPLE_WIDTH)
    data.adc_i_rms = audioop.rms(split_adc_data.current, config.SAMPLE_WIDTH)
    return data


def calculate_calibrated_power(split_adc_data, adc_rms, calibration):
    """
    Args:
        split_adc_data: Bunch with fields:
        - voltage (binary string): raw ADC data
        - current (binary string): raw ADC data

        adc_rms: Bunch with fields:
        - adc_v_rms (float)
        - adc_i_rms (float)
        
        calibration: Bunch with fields:
        - watts_per_adc_step (float)
        - volts_per_adc_step (float)
        - amps_per_adc_step (float)
        
    Returns:
        Bunch with fields:
        - real_power (float): watts
        - apparent_power (float): VA
        - volts_rms (float): volts
        - amps_rms (float): amps
        - power_factor (float)
        - frequency (float): voltage frequency in Hz
        - phase_diff (float or None): phase difference in degrees
            +ve : current leads voltage (capacitive)
            -ve : current lags voltage (inductive)
            None : unknown
    """
    
    data = Bunch()
    voltage, current = convert_adc_to_numpy_float(split_adc_data)
    voltage, current = shift_phase(voltage, current, calibration)
    
    # Frequency
    data.frequency = get_frequency(voltage)    
    
    # Real power
    inst_power = voltage * current # instantaneous power
    data.real_power = inst_power.mean() * calibration.watts_per_adc_step
    if data.real_power < 0:
        data.real_power = 0
    
    # Apparent power
    data.volts_rms = adc_rms.adc_v_rms * calibration.volts_per_adc_step
    data.amps_rms  = adc_rms.adc_i_rms * calibration.amps_per_adc_step
    data.apparent_power = data.volts_rms * data.amps_rms
    
    # Power factor
    data.power_factor = data.real_power / data.apparent_power
    
    # Is power factor leading (capacitive) or lagging (inductive)?
    try:
        data.phase_diff = (get_phase_diff(split_adc_data) -
                           calibration.phase_diff) / config.SAMPLES_PER_DEGREE
    except ZeroCrossingError as e:
        print(str(e), file=sys.stderr)
        data.phase_diff = None # unknown
    
    return data


def plot(voltage, current, calibration=None):
    """
    Plots voltage and current waveforms.  Does not do phase correction.
    
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
        except queue.Empty:
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


def indices_of_positive_peaks(data, frequency):
    """Returns the indices of the positive peaks for each cycle.
    
    Args:
        data (numpy array)
        frequency (float): sample frequency in Hz
    """
    n_samples_per_mains_cycle = config.FRAME_RATE / frequency 
    n_cycles = int(len(data) / n_samples_per_mains_cycle) # get floor
    n_samples_per_mains_cycle = int(round(n_samples_per_mains_cycle))
    
    indices_of_peaks = np.zeros(n_cycles)
    
    start_i = 0
    for cycle in range(n_cycles):
        end_i = (cycle+1)*n_samples_per_mains_cycle
        indices_of_peaks[cycle] = start_i + data[start_i:end_i].argmax()
        start_i = end_i
        
    return indices_of_peaks


def get_frequency(data):
    """
    Args:
        data (numpy array)
        
    Returns:
        float: frequency in Hz
    """
    pzc = positive_zero_crossings(data)
    mean_num_samples_per_cycle = np.diff(pzc).mean()
    freq = config.FRAME_RATE / mean_num_samples_per_cycle
    return freq


def get_phase_diff(split_adc_data, tolerance=config.PHASE_DIFF_TOLERANCE):
    """Finds the phase difference between the positive peaks
    of the voltage and current waveforms.
    
    Args:
        split_adc_data: Bunch with fields:
            - voltage (binary string): raw adc data
            - current (binary string): raw adc data

        tolerance (float): max number of samples by which 
            i and v zero crossings can differ.
    
    Returns:
        The mean number of samples by which the +ve peaks of the current
        and voltage waveforms differ.  
        Positive means current leads voltage ("leading" AKA capacitive)
        Negative means current lags voltage ("lagging" AKA inductive)
        http://sg.answers.yahoo.com/question/index?qid=20111107044946AACm7c8
    """ 

    voltage, current = convert_adc_to_numpy_float(split_adc_data)
    FREQ = get_frequency(voltage)
    v_peaks = indices_of_positive_peaks(voltage, FREQ)
    i_peaks = indices_of_positive_peaks(current, FREQ)
    
    # sanity check length
    if not (len(i_peaks)-2 < len(v_peaks) < len(i_peaks)+2):
        raise ZeroCrossingError("ERROR: number of current peaks ({})"
                                " too dissimilar to\n"
                                "       number of voltage peak ({})."
                                .format(len(i_peaks), len(v_peaks)))

    
    # go through each peak in turn and compare I with V
    i_offset = 0
    phase_diffs = []
    for i in range(len(v_peaks)):
        try:
            phase_diff = i_peaks[i+i_offset] - v_peaks[i]
        except IndexError:
            continue
        else:
            if abs(phase_diff) < tolerance: # Then this is a valid comparison            
                phase_diffs.append(phase_diff)
            elif phase_diff > tolerance:
                i_offset -= 1
            else:
                i_offset += 1

    mean_phase_diff = np.mean(phase_diffs)
    std_phase_diff = np.std(phase_diffs)
            
    print("Raw, uncalibrated phase difference (+ve means I leads V):" )
    print("  phase diff mean samples = {:.1f}, mean degrees = {:.2f}\n"
          "  phase diff std samples  = {:.3f},  std degrees = {:.2f}\n"
          "  phase diff number of valid comparisons = {}"
          .format(mean_phase_diff, mean_phase_diff / config.SAMPLES_PER_DEGREE,
                  std_phase_diff, std_phase_diff / config.SAMPLES_PER_DEGREE,
                  len(phase_diffs)))
    
    return mean_phase_diff


def load_calibration_file(calibration_parser=None):
    """Loads config.CALIBRATION_FILENAME.
    
    Returns a Bunch with fields:
        - volts_per_adc_step (float)
        - amps_per_adc_step (float)
        - phase_diff (float)
        - watts_per_adc_step (float)
    """
    if calibration_parser is None:
        calibration_parser = ConfigParser.RawConfigParser()
        calibration_parser.read(config.CALIBRATION_FILENAME)
        
    calib = Bunch()
    calib_section = "Calibration"
    try:
        calib.volts_per_adc_step = calibration_parser.getfloat(calib_section, 
                                                          "volts_per_adc_step")
        calib.amps_per_adc_step = calibration_parser.getfloat(calib_section,
                                                         "amps_per_adc_step")
        calib.phase_diff = calibration_parser.getfloat(calib_section,
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


def print_power(calcd_data, wu_data=None):
    def diff(a, b):
        return abs(a-b) / max(a,b)
    
    if calcd_data.phase_diff is None:
        leading_or_lagging = "unknown"
    else:
        leading_or_lagging = ("I leads V (capacitive)"
                              if calcd_data.phase_diff > 0
                              else "I lags V (inductive)")
        leading_or_lagging += " {:.3f} degrees".format(calcd_data.phase_diff)
    
    print("         VOLTS  |   AMPS |     REAL |  APPARENT |  PF ")
    print("   SCPM: {:>6.1f} | {:>6.3f} | {:>8.1f} |  {:>8.2f} | {:>3.2f} {}"
          .format(calcd_data.volts_rms, calcd_data.amps_rms, 
                  calcd_data.real_power, calcd_data.apparent_power, 
                  calcd_data.power_factor, leading_or_lagging))
    
    if wu_data is not None:
        print("WattsUp: {:>6.1f} | {:>6.3f} | {:>8.1f} |  {:>8.2f} | {:>3.2f}"
              .format(wu_data.volts_rms, wu_data.amps_rms, 
                      wu_data.real_power, wu_data.apparent_power, 
                      wu_data.power_factor))
        
        print("   Diff:{:>7.3%} |{:>7.3%} |  {:>7.3%} |   {:>7.3%} |{:>7.3%}"
              .format(diff(calcd_data.volts_rms, wu_data.volts_rms),
                      diff(calcd_data.amps_rms, wu_data.amps_rms),
                      diff(calcd_data.real_power, wu_data.real_power),
                      diff(calcd_data.apparent_power, wu_data.apparent_power),
                      diff(calcd_data.power_factor, wu_data.power_factor)))

    print("SCPM voltage frequency: {:.3f}".format(calcd_data.frequency))


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
    
    calibration_parser = ConfigParser.RawConfigParser()
    calibration_parser.read(config.CALIBRATION_FILENAME)
    try:
        calibration_parser.add_section("Calibration")
    except ConfigParser.DuplicateSectionError:
        print("Overwriting existing calibration file",
              config.CALIBRATION_FILENAME)
        calib = load_calibration_file(calibration_parser)
    else:
        calib = Bunch()
    
    while True:
        wu_data = wu.get() # blocking
        if wu_data is None:
            continue
        
        time.sleep(config.RECORD_SECONDS)
        
        # Now get ADC data recorded at wu_data.time
        adc_data = find_time(adc_data_queue, wu_data.time)
                
        if adc_data:
            split_adc_data = split_channels(adc_data.data)
            adc_rms = calculate_adc_rms(split_adc_data)
            
            # Voltage
            n_v_samples += 1
            v_acumulator += wu_data.volts_rms / adc_rms.adc_v_rms
            calib.volts_per_adc_step = v_acumulator / n_v_samples
            
            # Current           
            if wu_data.amps_rms > 0.1:
                n_i_samples += 1 
                i_acumulator += wu_data.amps_rms / adc_rms.adc_i_rms
                calib.amps_per_adc_step = i_acumulator / n_i_samples
                calib.watts_per_adc_step = (calib.amps_per_adc_step *
                                            calib.volts_per_adc_step)
                
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
                        calib.phase_diff = pd_acumulator / n_pd_samples
                        print("Average phase diff =", calib.phase_diff)
                        calibration_parser.set("Calibration", "phase_difference",
                                          calib.phase_diff)
                else:
                    processing_pf = False

                print("Watts up reports power factor "
                      "is {:2.2f} so".format(wu_data.power_factor),
                      "will" if processing_pf else "will not",
                      "calibrate phase angle...")

            else:
                print("Not sampling amps because the WattsUp reading too low.")    
                        
            calcd_data = calculate_calibrated_power(split_adc_data, 
                                                    adc_rms, calib)
            
            print_power(calcd_data, wu_data)
            
            print("adc time = {}, watts up time = {}, time diff = {:1.3f}s \n"
                  .format(adc_data.time, wu_data.time,
                          adc_data.time - wu_data.time))

            calibration_parser.set("Calibration", "volts_per_adc_step",
                              calib.volts_per_adc_step)
            calibration_parser.set("Calibration", "amps_per_adc_step",
                              calib.amps_per_adc_step)
            with open(config.CALIBRATION_FILENAME, "wb") as calibfile:
                calibration_parser.write(calibfile)
        else:
            print("Could not find match for", wu_data.time, file=sys.stderr)
            

    
def get_wavfile_name(t):
    return config.FLAC_FILENAME_PREFIX + "-{:f}".format(t) + ".wav" 


def get_wavfile(wavefile_name):
    wavfile = wave.open(wavefile_name, 'wb')
    wavfile.setnchannels(config.N_CHANNELS)
    wavfile.setsampwidth(config.SAMPLE_WIDTH)
    wavfile.setframerate(config.FRAME_RATE)
    return wavfile    

