#! /usr/bin/python

""" 
THIS CODE IS JUST ME LEARNING TO USE ALSAAUDIO.
IT DOESN'T YET DO ANY POWER MEASUREMENT STUFF! 

Requirements
------------

  Ubuntu packages
     sudo apt-get install python-dev python-pip alsa alsa-tools libasound2-dev

  pyalsaaudio
     * Install on Linux using "sudo pip install pyalsaaudio"
                           or "sudo easy_install pyalsaaudio".

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
import pyaudio # docs: http://people.csail.mit.edu/hubert/pyaudio/
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
from threading import Thread

try:
    from Queue import Queue, Empty
except ImportError:
    from queue import Queue, Empty # python 3.x

# Load configuration file
config = None
VOLTS_PER_ADC_STEP = None
AMPS_PER_ADC_STEP = None
PHASE_DIFF = None

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

CHUNK = 1024
FORMAT = pyaudio.paInt32
CHANNELS = 2
RATE = 96000 #Hz
RECORD_SECONDS = 1
WAV_FILENAME = "voltage_current"
DOWNSAMPLED_RATE = 15000 # Hz

print("VOLTS_PER_ADC_STEP =", VOLTS_PER_ADC_STEP)
print("AMPS_PER_ADC_STEP =", AMPS_PER_ADC_STEP)
print("WATTS_PER_ADC_STEP =", WATTS_PER_ADC_STEP)

p = pyaudio.PyAudio()
audio_stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                      input=True, frames_per_buffer=CHUNK)

WIDTH = p.get_sample_size(FORMAT)

# Named tuples
TVI = collections.namedtuple('TVI', ['time', 'voltage', 'current'])
Tdata = collections.namedtuple('Tdata', ['time', 'data'])
Power = collections.namedtuple('Power', ['time', 'real', 'apparent', 'v_rms'])


def run_command(cmd):
    """Run a UNIX shell command.
    
    Args:
        cmd (list of strings)
    """
    try:
        p = subprocess.Popen(cmd, stderr=subprocess.PIPE)
        p.wait()
    except Exception, e:
        print("ERROR: Failed to run '{}'".format(" ".join(cmd)), file=sys.stderr)
        print("ERROR:", str(e), file=sys.stderr)
    else:
        if p.returncode == 0:
            print("Successfully ran '{}'".format(" ".join(cmd)))
        else:
            print("ERROR: Failed to run '{}'".format(" ".join(cmd)), file=sys.stderr)
            print(p.stderr.read(), file=sys.stderr)


def config_mixer():
    print("Configuring mixer...")
    run_command(["amixer", "sset", "Input Source", "Rear Mic"])
    run_command(["amixer", "set", "Digital", "60", "capture"])
    run_command(["amixer", "set", "Capture", "16", "capture"])
    

def get_adc_data():
    """
    Get data from the sound card's analogue to digital converter (ADC).
    
    Returns:
        A Tdata named tuple with fields:
        - time (float): UNIX timestamp immediately prior to sampling
        - data (binary string): stereo ADC data
    """
    t = time.time()
    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        for retry in range(5):
            try:
                data = audio_stream.read(CHUNK)
            except IOError, e:
                print("ERROR: ", str(e), file=sys.stderr)
            else:
                frames.append(data)
                break

    stereo = b''.join(frames)
    return Tdata(t, stereo)


def split_channels(tdata):
    """
    Args:
        tdata: a Tdata named tuple with fields:
        - time (float): UNIX timestamp
        - data (binary string): stereo raw ADC data
    
    Returns:
        A TVI named tuple with fields:
        - t (float): UNIX timestamp immediately prior to sampling
        - voltage (binary string): raw ADC data 
        - current (binary string): raw ADC data
    """
    voltage = audioop.tomono(tdata.data, WIDTH, 1, 0)
    current = audioop.tomono(tdata.data, WIDTH, 0, 1)
    return TVI(tdata.time, voltage, current)


def enqueue_adc_data(adc_data_queue):
    """This will be run as a separate thread.""" 
    while True:
        adc_data_queue.put(get_adc_data())


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
        split_adc_data: TVI named tuple with fields time, voltage and current.
    """
    voltage, current = convert_adc_to_numpy_float(split_adc_data)
    voltage, current = shift_phase(voltage, current)
    
    inst_power = voltage * current # instantaneous power
    real_power = inst_power.mean() * WATTS_PER_ADC_STEP
    if real_power < 0:
        real_power = 0
    
    v_rms = audioop.rms(split_adc_data.voltage, WIDTH)
    i_rms = audioop.rms(split_adc_data.current, WIDTH)
    apparent_power = v_rms * i_rms * WATTS_PER_ADC_STEP
    
    power_factor = real_power / apparent_power
    
    # TODO: leading / lagging phase
    print("real power = {:4.2f}W, apparent_power = {:4.2f}VA, "
          "power factor = {:1.3f}, v_rms = {:4.2f}V, i_rms = {:4.4f}A"
          .format(real_power, apparent_power, power_factor,
                  v_rms * VOLTS_PER_ADC_STEP, 
                  i_rms * AMPS_PER_ADC_STEP))
    print("raw v_rms =", v_rms, ", raw i_rms = ", i_rms)
    
    return Power(split_adc_data.time, real_power, apparent_power, v_rms)
  

def plot(voltage, current):
    """
    Args:
        voltage, current (numpy arrays)
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

    return args

def start_adc_data_queue_and_thread():
    adc_data_queue = Queue()
    adc_thread = Thread(target=enqueue_adc_data, args=(adc_data_queue, ))
    adc_thread.daemon = True # this thread dies with the program
    adc_thread.start()
    
    return adc_data_queue, adc_thread    
    
def get_wavfile_name(t):
    return WAV_FILENAME + "-{:f}".format(t) + ".wav" 
    
def get_wavfile(wavefile_name):
    wavfile = wave.open(wavefile_name, 'wb')
    wavfile.setnchannels(CHANNELS)
    wavfile.setsampwidth(WIDTH)
    wavfile.setframerate(DOWNSAMPLED_RATE)
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
        p = None
        cmd = ""
        while True:
            try:
                adc_data = adc_data_queue.get()
                split_adc_data = split_channels(adc_data)
                power = calculate_power(split_adc_data)
                # TODO: do save power to disk
                
                # WAV dump to disk
                downsampled, filter_state = audioop.ratecv(adc_data.data, 
                                                           WIDTH, CHANNELS, 
                                                           RATE,
                                                           DOWNSAMPLED_RATE,
                                                           filter_state)
                
                t = datetime.datetime.fromtimestamp(adc_data.time)
                # TODO: change to hourly recordings after testing!
                
                # Check if it's time to create a new data file
                if wavfile is None or t.minute == (prev.minute+1)%60:
                    if wavfile is not None:
                        wavfile.close()
                        
                        # Check if the previous conversion process has completed
                        if p is not None:
                            p.poll()
                            if p.returncode is None:
                                print("WARNING: command has not terminated yet:",
                                       cmd, file=sys.stderr)
                            elif p.returncode == 0:
                                print("Previous conversion successfully completed.")
                            else:
                                print("WARNING: Previous conversion FAILED.",
                                       cmd, file=sys.stderr)
                                
                        # Run new shell process to compress wav file
                        base_filename = wavfile_name.rpartition('.')[0]
                        cmd = "ffmpeg -i {filename}.wav -acodec pcm_s24le {filename}_24bit.wav"\
                              " && flac {filename}_24bit.wav -o {filename}.flac --verify --best"\
                              " && rm -f {filename}_24bit.wav {filename}.wav"\
                              .format(filename=base_filename)
                        print("Running", cmd)                                
                        p = subprocess.Popen(cmd, shell=True)
                        
                    wavfile_name = get_wavfile_name(adc_data.time)
                    wavfile = get_wavfile(wavfile_name)
                    
                prev = t
                
                wavfile.writeframes(downsampled)


            except KeyboardInterrupt:
                wavfile.close()
                audio_stream.stop_stream()
                audio_stream.close()
                p.terminate()
                break
        
    print("")

if __name__=="__main__":
    main()
