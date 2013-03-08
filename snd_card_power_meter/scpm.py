#! /usr/bin/python

""" 
THIS CODE IS JUST ME LEARNING TO USE ALSAAUDIO.
IT DOESN'T YET DO ANY POWER MEASUREMENT STUFF! 

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
  
Calibration with a Watts Up
---------------------------

NEW CALIBRATION TECHNIQUE:
    Plug in a WattsUp
        run ./scpm --calibrate
    This will update (or create) a config.cfg file with the calibration
    parameters.

OLD CALIBRATION TECHNIQUE:

Log data from Watts Up using
   screen -dmL ./wattsup ttyUSB0 volts
  
"""

from __future__ import print_function, division
import numpy as np
import wave
import subprocess
import sys
import argparse
import audioop # docs: http://docs.python.org/2/library/audioop.html
import time
import wattsup
import collections
import os
import ConfigParser # docs: http://docs.python.org/2/library/configparser.html
import datetime
import sampler
from struct import Struct

# Load configuration file
config = None
VOLTS_PER_ADC_STEP = None
AMPS_PER_ADC_STEP = None
PHASE_DIFF = None

def load_config():
    try:
        # If we load this script from an interactive Python interpreter for
        # testing then __file__ won't evaluate correctly.
        CONFIG_FILE = os.path.dirname(__file__) + "/../config.cfg"
    except NameError:
        pass
    else:
        config = ConfigParser.RawConfigParser()
        config.read(CONFIG_FILE)
        try:
            config.add_section("Calibration")
        except ConfigParser.DuplicateSectionError:
            try:
                VOLTS_PER_ADC_STEP = config.getfloat("Calibration", "volts_per_adc_step")
                AMPS_PER_ADC_STEP = config.getfloat("Calibration", "amps_per_adc_step")
                PHASE_DIFF = config.getfloat("Calibration", "phase_difference")
            except ConfigParser.NoOptionError:
                pass
    
    if VOLTS_PER_ADC_STEP and AMPS_PER_ADC_STEP:
        WATTS_PER_ADC_STEP = VOLTS_PER_ADC_STEP * AMPS_PER_ADC_STEP
    else:
        WATTS_PER_ADC_STEP = None
    
    print("VOLTS_PER_ADC_STEP =", VOLTS_PER_ADC_STEP)
    print("AMPS_PER_ADC_STEP  =", AMPS_PER_ADC_STEP)
    print("WATTS_PER_ADC_STEP =", WATTS_PER_ADC_STEP)

DOWNSAMPLED_RATE = 16000 # Hz (MIT REDD uses 15kHz but 16kHz is a standard
#                              rate and so increases compatibility)
WAV_FILENAME = "voltage_current"


# Named tuples
TVI = collections.namedtuple('TVI', ['time', 'voltage', 'current'])
Power = collections.namedtuple('Power', ['time', 'real', 'apparent', 'v_rms'])

        
def split_channels(stereo):
    """
    Args:
        stereo (binary string): stereo raw ADC data
    
    Returns:
        A Struct with fields:
        - voltage (binary string): raw ADC data 
        - current (binary string): raw ADC data
    """
    data = Struct()
    data.voltage = audioop.tomono(stereo, sampler.WIDTH, 1, 0)
    data.current = audioop.tomono(stereo, sampler.WIDTH, 0, 1)
    return data


def convert_adc_to_numpy_float(adc_data):
    """Convert from binary string to numpy array.
    
    Args:
        adc_data: a TVI name tuple with fields:
        - voltage (binary string): raw voltage data from ADC
        - current (binary string): raw current data from ADC
        
    Returns:
        voltage, current (each a float64 numpy vector)
    """
    voltage = np.fromstring(adc_data.voltage, dtype='int32').astype(np.float64)
    current = np.fromstring(adc_data.current, dtype='int32').astype(np.float64)
    
    return voltage, current    


def shift_phase(voltage, current):
    """Shift voltage and current by PHASE_DIFF number of samples.
    The aim is to correct for phase errors caused by the measurement system.
    
    Args:
        voltage, current (numpy arrays)
    """
    if PHASE_DIFF is None:
        return voltage, current
    
    pd = abs(int(round(PHASE_DIFF)))
    if pd == 0:
        pass # do nothing if the phase difference is zero
    elif PHASE_DIFF < 0: # current leads voltage
        voltage = voltage[pd:]
        current = current[:-pd]
    else:
        voltage = voltage[:-pd]
        current = current[pd:]
        
    return voltage, current


def calculate_power(split_adc_data):
    """
    Args:
        split_adc_data: TVI named tuple with fields:
        - time (float): UNIX timestamp
        - voltage, current (binary string with raw ADC data)
        
    Returns:
        Power named tuple with fields:
        - time (float): UNIX timestamp
        - real_power (float)
        - apparent_power (float)
        - v_rms (float)
    """
    voltage, current = convert_adc_to_numpy_float(split_adc_data)
    voltage, current = shift_phase(voltage, current)
    
    d = Data()
    d.time = split_adc_data.time
    
    inst_power = voltage * current # instantaneous power
    d.real_power = inst_power.mean() * WATTS_PER_ADC_STEP
    if d.real_power < 0:
        d.real_power = 0
    
    d.v_rms = audioop.rms(split_adc_data.voltage, WIDTH)
    i_rms = audioop.rms(split_adc_data.current, WIDTH)
    d.apparent_power = d.v_rms * i_rms * WATTS_PER_ADC_STEP
    
    power_factor = d.real_power / d.apparent_power
    
    # TODO: leading / lagging phase
    print("real power = {:4.2f}W, apparent_power = {:4.2f}VA, "
          "power factor = {:1.3f}, v_rms = {:4.2f}V, i_rms = {:4.4f}A"
          .format(d.real_power, d.apparent_power, power_factor,
                  d.v_rms * VOLTS_PER_ADC_STEP, 
                  i_rms * AMPS_PER_ADC_STEP))
    print("raw v_rms =", d.v_rms, ", raw i_rms = ", i_rms)
    
    return d
  

def plot(voltage, current):
    """
    Args:
        voltage, current (numpy arrays containing raw ADC values)
    """
    import matplotlib.pyplot as plt
        
    if VOLTS_PER_ADC_STEP:
        print("VOLS_PER_ADC_STEP =", VOLTS_PER_ADC_STEP)        
        v_unit = "v"
        voltage *= VOLTS_PER_ADC_STEP
    else:
        v_unit = " raw ADC float"
        
    if AMPS_PER_ADC_STEP:
        print("AMPS_PER_ADC_STEP =", AMPS_PER_ADC_STEP)
        i_unit = "A"
        current *= AMPS_PER_ADC_STEP
    else:
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

def get_phase_diff(split_adc_data):
    """Finds the phase difference between the positive-going zero crossings
    of the voltage and current waveforms.
    
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
    TOLERANCE = RATE / 100 # max number of samples by which i and v zero crossings can differ
    i_offset = 0
    phase_diffs = []
    for i in range(len(vzc)):
        try:
            phase_diff = izc[i+i_offset] - vzc[i]
        except IndexError:
            continue
        
        if abs(phase_diff) < TOLERANCE: # This is a valid comparison            
            phase_diffs.append(phase_diff)
        elif phase_diff > TOLERANCE:
            i_offset -= 1
        else:
            i_offset += 1
            
    print("phase diff = {:.1f} samples, std = {:.3f}".format(np.mean(phase_diffs),
                                                         np.std(phase_diffs)))
    
    return np.mean(phase_diffs)
    

def calibrate(adc_data_queue):    
    wu = wattsup.WattsUp()
    
    v_acumulator = np.float64(0.0) # accumulator for voltage
    i_acumulator = np.float64(0.0) # accumulator for current
    pd_acumulator = np.float64(0.0) # accumulator for phase difference
    n_v_samples = 0 # number of voltage samples
    n_i_samples = 0 # number of current samples
    n_pd_samples = 0 # number of phase diff samples
    av_i_calibration = np.float64(0.0)
    av_v_calibration = np.float64(0.0)
    adc_i_rms = 0.0
    av_pd = np.float64(0.0) # average phase diff
    
    while True:
        wu_data = wu.get() # blocking
        
        # Now get ADC data recorded at wu_data.time
        adc_data = find_time(adc_data_queue, wu_data.time)
        split_adc_data = split_channels(adc_data)
        
        if split_adc_data:
            # Voltage
            adc_v_rms = audioop.rms(split_adc_data.voltage, p.get_sample_size(FORMAT)) 
            n_v_samples += 1
            v_acumulator += wu_data.volts / adc_v_rms
            av_v_calibration = v_acumulator / n_v_samples # average voltage calibration
            
            # Current
            adc_i_rms = audioop.rms(split_adc_data.current, p.get_sample_size(FORMAT))            
            if wu_data.amps > 0.1:
                n_i_samples += 1 
                i_acumulator += wu_data.amps / adc_i_rms
                av_i_calibration = i_acumulator / n_i_samples # average voltage calibration
                
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
                        config.set("Calibration", "phase_difference", av_pd)
                else:
                    processing_pf = False

                print("Watts up reports power factor "
                      "is {:2.2f} so".format(wu_data.power_factor),
                      "will" if processing_pf else "will not",
                      "calibrate phase angle...")

                
            else:
                print("Not sampling amps because the WattsUp reading is too low.")    
            
            print("WattsUp:    volts = {:>03.2f}, amps = {:>02.2f}, \n"
                  "Calculated: volts = {:>03.2f}, amps = {:>02.2f}, \n"
                  "Raw ADC: voltage = {}, current = {}, \n"
                  "v_calibration = {}, i_calibration = {}, \n"
                  "adc_data.time = {}, wu_data.time = {}, time diff = {:1.3f}s \n"
                  .format(wu_data.volts, wu_data.amps,
                          adc_v_rms * av_v_calibration,
                          adc_i_rms * av_i_calibration,
                          adc_v_rms, adc_i_rms,
                          av_v_calibration, av_i_calibration,
                          adc_data.time, wu_data.time, adc_data.time - wu_data.time))

            config.set("Calibration", "volts_per_adc_step", av_v_calibration)
            config.set("Calibration", "amps_per_adc_step", av_i_calibration)
            with open(CONFIG_FILE, "wb") as configfile:
                config.write(configfile)
        else:
            print("Could not find match for", wu_data.time, file=sys.stderr)
            time.sleep(RECORD_SECONDS*2)
            
        # TODO:
        #  * do calcs for amps        


def setup_argparser():
    # Process command line _args
    parser = argparse.ArgumentParser(description="Record voltage and current"
                                     "  waveforms using the sound card.")
       
    parser.add_argument("--no-mixer-config", dest="config_mixer", 
                        action="store_false",
                        help="Don't modify ALSA mixer (i.e. use existing system settings)")
    
    parser.add_argument("--calibrate", 
                        action="store_true",
                        help="Run calibration using a WattsUp meter")
    
    parser.add_argument("--plot", 
                    action="store_true",
                    help="Plot one second's worth of data")

    
    args = parser.parse_args()
    
    if WATTS_PER_ADC_STEP is None and not args.calibrate:
        print("Could not load config file.  Please run with --calibrate option"
              " to generate a config file.", file=sys.stderr)
        sys.exit(1)

    return args

    
def get_wavfile_name(t):
    return WAV_FILENAME + "-{:f}".format(t) + ".wav" 
    
def get_wavfile(wavefile_name):
    wavfile = wave.open(wavefile_name, 'wb')
    wavfile.setnchannels(CHANNELS)
    wavfile.setsampwidth(WIDTH)
    wavfile.setframerate(RATE)
    return wavfile    

def main():
    args = setup_argparser()
    
    if args.config_mixer:
        config_mixer()

    adc_data_queue, adc_thread = start_adc_data_queue_and_thread()

    if args.calibrate:
        calibrate(adc_data_queue)
    elif args.plot:
        adc_data = adc_data_queue.get()
        voltage, current = convert_adc_to_numpy_float(adc_data)
        plot(voltage, current)
    else:
        filter_state = None
        wavfile = None
        proc = None # a subprocess.Popen object
        cmd = ""
        while True:
            try:
                adc_data = adc_data_queue.get()
                split_adc_data = split_channels(adc_data)
                power = calculate_power(split_adc_data)
                # TODO: do save power to disk
                
                # WAV dump to disk
                # Check if it's time to create a new data file
                t = datetime.datetime.fromtimestamp(adc_data.time)
                if wavfile is None or t.minute == (prev.minute+1)%60:
                    # TODO: change to hourly recordings after testing!
                    if wavfile is not None:
                        wavfile.close()
                        
                        # Check if the previous conversion process has completed
                        if proc is not None:
                            proc.poll()
                            if proc.returncode is None:
                                print("WARNING: command has not terminated yet:",
                                       cmd, file=sys.stderr)
                            elif proc.returncode == 0:
                                print("Previous conversion successfully completed.")
                            else:
                                print("WARNING: Previous conversion FAILED.",
                                       cmd, file=sys.stderr)
                                
                        # Run new shell process to compress wav file
                        base_filename = wavfile_name.rpartition('.')[0]
                        # sox is a high quality audio conversion program
                        # the -v -L rate option forced very high quality
                        # with linear phase.
                        cmd = "sox --no-dither {filename}.wav --bits 24"\
                              " --compression 8 {filename}.flac"\
                              " rate -v -L {downsampled_rate}"\
                              " && rm {filename}.wav"\
                              .format(filename=base_filename,
                                      downsampled_rate=DOWNSAMPLED_RATE)

                        print("Running", cmd)                                
                        proc = subprocess.Popen(cmd, shell=True)
                        
                    wavfile_name = get_wavfile_name(adc_data.time)
                    wavfile = get_wavfile(wavfile_name)
                    
                prev = t
                
                wavfile.writeframes(adc_data.data)


            except KeyboardInterrupt:
                wavfile.close()
                audio_stream.stop_stream()
                audio_stream.close()
                proc.terminate()
                break
        
    print("")

if __name__=="__main__":
    main()
