#!/usr/bin/env python

# Make a few thousand tf data files
#  for training the VAE models.


# This script does not run the commands - it makes a list of commands
#  (in the file 'run_training_tensor_{args.month}.sh').

import os
import datetime

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--month", help="Month", type=int, required=True)
args = parser.parse_args()

# Function to check if the job is already done for this timepoint
def is_done(year, month, day, group):
    op_file_name = (
        ("%s/Proxy_20CR/datasets/ERA5/daily_T850/regridded_version/" + "%s/%04d-%02d-%02d.tfd")
    ) % (
        os.getenv("SCRATCH"),
        group,
        year,
        month,
        day,
    )
    if os.path.isfile(op_file_name):
        return True
    return False

full_path = os.path.realpath(__file__)
print(os.path.dirname(full_path))

f = open(f"run_training_tensor_{args.month}.sh", "w+")
f.write("#!/bin/bash\n")


count = 1
for year in range(1979, 2022+1):
    start_day = datetime.date(year, args.month, 1)
    if args.month < 12:
        end_day = datetime.date(year, args.month+1, 1)
    else:
        end_day = datetime.date(year+1, 1, 1)

    current_day = start_day
    while current_day < end_day:
        training_section = count%10 # this wont affect data for years 2015-2022 that go to validation and test sets
        if not is_done(
            current_day.year,
            current_day.month,
            current_day.day,
            "training%d" % training_section,
        ):
            cmd = ("python %s/make_training_tensor_day.py --year=%d --month=%d --day=%d --training_section=%d --experiment=3\n") % (
                os.path.dirname(full_path),
                current_day.year,
                current_day.month,
                current_day.day,
                training_section,
            )
            f.write(cmd)
        current_day = current_day + datetime.timedelta(days=1)
        count += 1

f.close()
