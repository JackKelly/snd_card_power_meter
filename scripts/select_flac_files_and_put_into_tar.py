#! /usr/bin/python

from __future__ import print_function, division
import argparse
from os.path import isdir, join
from os import listdir, mkdir
from subprocess import call

# Read command line arguments
parser = argparse.ArgumentParser(description=
    "Select a set of SCPM flac files.")
parser.add_argument("src_dir", help="source directory", type=str)
parser.add_argument("dst_filename", help="destination filename", type=str)
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
    selected_files = [join(args.src_dir, filename) for filename in selected_files]

    # create dst_dir if necessary
    cmd = ("tar -cvf {:s} {:s}"
           .format(args.dst_filename, " ".join(selected_files)))
    print("Running", cmd, "...")
    call(cmd, shell=True)
    print("Done!")


if __name__ == "__main__":
    main()
