#! /usr/bin/python
"""
Plot data from a mains.dat file.

"""

from __future__ import print_function, division
import argparse, logging, datetime
import numpy as np
import matplotlib.dates
import matplotlib.pyplot as plt
log = logging.getLogger("scpm")


def setup_argparser():
    # Process command line _args
    parser = argparse.ArgumentParser(description="Load and plot power data "
                                                 "from SCPM .dat file.")
       
    parser.add_argument('input_file', help='FLAC file to read. NO suffix!')

    args = parser.parse_args()
    return args


def get_data_from_file(input_file):
    return np.genfromtxt(input_file, delimiter=' ', usecols=(0,1,2,3),
                         names=('timestamp', 'active_power',
                                'apparent_power', 'volts'))

def smooth(input_vector, timestamps, factor):
    """
    Args:
        - input_vector (Numpy array)
        - timestamps (Numpy array of UNIX timestamps)
        - factor (int): the number of rows in input_vector to average
          together in the output
    Returns:
        a Numpy array with length = input_vector.size / factor
    """
    assert(input_vector.size == timestamps.size)
    
    downsampled_length = int(np.floor(input_vector.size / factor))
    downsampled = np.empty([downsampled_length,2])
    
    for i in range(downsampled_length):
        downsampled[i,0] = timestamps[(i+1)*factor]  
        downsampled[i,1] = input_vector[(i*factor):((i+1)*factor)].mean()

    return downsampled


def main():
    args = setup_argparser()
    data = get_data_from_file(args.input_file)
    x = [datetime.datetime.fromtimestamp(t) for t in data['timestamp']]
    
    fig = plt.figure()
    ax1 = fig.add_subplot(3,1,1)
    
    ###########################
    # Plot raw 1-second data
    ax1.plot(x, data['active_power'], label="active power (watts)")
    ax1.set_title('1 second sample period')
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax1.plot(x, data['apparent_power'], label="apparent power (VA)")
    #ax1.plot(x, data['volts'], label="volts")
    
    ###########################
    # 5-second downsample
    downsampled5 = smooth(data['active_power'], data['timestamp'], 5)
    x_downsampled = [datetime.datetime.fromtimestamp(t) 
                     for t in downsampled5[:,0]]
    ax2 = fig.add_subplot(3,1,2, sharex=ax1, sharey=ax1)
    ax2.plot(x_downsampled, downsampled5[:,1], label='downsampled5')
    ax2.set_title('5 second sample period')
    ax2.set_ylabel('active power (watts)')    
    plt.setp(ax2.get_xticklabels(), visible=False) 
    
    ######################
    # 10-second downsample
    downsampled10 = smooth(data['active_power'], data['timestamp'], 10)
    x_downsampled = [datetime.datetime.fromtimestamp(t) 
                     for t in downsampled10[:,0]]
    ax3 = fig.add_subplot(3,1,3, sharex=ax1, sharey=ax1)
    ax3.plot(x_downsampled, downsampled10[:,1], label='downsampled10')
    ax3.set_title('10 second sample period')
    
    date_formatter = matplotlib.dates.DateFormatter("%d/%m\n%H:%M:%S")
    ax3.xaxis.set_major_formatter( date_formatter )     
   
    plt.show()

if __name__ == '__main__':
    main()
