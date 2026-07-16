#!/usr/bin/env python

# Make an ER5 T850 variability climatology file for each day

# This script does not run the commands - it makes a list of commands
#  (in the file 'run_variability_climatology_{args.month}.sh').

import os
import datetime

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--month", help="Integer month", type=int, required=True)
args = parser.parse_args()

# Function to check if the job is already done for this timepoint
def is_done(month, day):
    op_file_name = (
        ("%s/Proxy_20CR/datasets/ERA5/daily_T850/regridded_version/variability_climatology/%02d/%02d.nc")
    ) % (
        os.getenv("SCRATCH"),
        month,
        day,
    )
    if os.path.isfile(op_file_name):
        return True
    return False


full_path = os.path.realpath(__file__)
print(os.path.dirname(full_path))

f = open(f"run_variability_climatology_{args.month}.sh", "w+")
f.write("#!/bin/bash\n")

start_day = datetime.date(1981, args.month, 1)
if args.month < 12:
    end_day = datetime.date(1981, args.month+1, 1)
else:
    end_day = datetime.date(1982, 1, 1)

current_day = start_day
while current_day < end_day:
    if not is_done(
        current_day.month,
        current_day.day,
    ):
        cmd = ("python %s/make_variability_climatology_day.py --month=%d --day=%d \n") % (
            os.path.dirname(full_path),
            current_day.month,
            current_day.day,
        )
        f.write(cmd)
    current_day = current_day + datetime.timedelta(days=1)

f.close()
