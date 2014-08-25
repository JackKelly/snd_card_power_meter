#! /usr/bin/python

from __future__ import print_function, division
import subprocess
import argparse
from os.path import isdir, join
from os import listdir, mkdir


# Read command line arguments
parser = argparse.ArgumentParser(description=
    "Convert a set of SCPM flac files to a different sample rate.")
parser.add_argument("src_dir", help="source directory", type=str)
parser.add_argument("dst_dir", help="destination directory", type=str)
parser.add_argument("rate", help="New sample rate in Hz (int)", type=int)
parser.add_argument("start", help="Start unix timestamp (int)", type=int)
parser.add_argument("end", help="End unix timestamp (int)", type=int)
args = parser.parse_args()


def main():
    # check source directory is valid
    if not isdir(args.src_dir):
        raise RuntimeError("'{}' is not a valid directory".format(args.src_dir))

    # read all flac files in src_dir
    all_flac_filenames = [filename for filename in listdir(args.src_dir)
                          if filename.endswith('.flac')]

    # create list of (filename, timestamp) tuples
    filenames_and_timestamps = [(filename, int(filename[3:13]))
                                for filename in all_flac_filenames]

    # select files between args.start and args.end
    selected_files = [filename for filename, timestamp 
                      in filenames_and_timestamps
                      if args.start <= timestamp <= args.end]

    print("Selected {:d} files".format(len(selected_files)))

    # create dst_dir if necessary
    if not isdir(args.dst_dir):
        mkdir(args.dst_dir)

    # convert each file
    for filename in selected_files:
        convert_sample_rate(filename)


def convert_sample_rate(filename):
    # Run new sox process to compress wav file                
    # sox is a high quality audio conversion program
    # the "rate -v -L" option forces very high quality
    # rate conversion with linear phase.
    cmd = ("sox --no-dither {src_filename}"
           " --compression 8 {dst_filename}"
           " rate -v -L {rate}"
           .format(filename=filename, 
                   src_filename=join(args.src_dir, filename),
                   dst_filename=join(args.dst_dir, filename),
                   rate=args.rate))

    print("Running: " + cmd)
    subprocess.call(cmd, shell=True)

if __name__ == "__main__":
    main()
