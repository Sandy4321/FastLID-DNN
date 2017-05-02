import numpy as np
import time

import sys

if len(sys.argv) != 2:
    print "Usage: python convergence.py <log file>"
    sys.exit(1)

log_filename = sys.argv[1]

# Read in job times
jobs = dict()
with open(log_filename, 'r') as log:
    for log_line in log.readlines():
        key_phrase = "        "
        if key_phrase in log_line:
            job_id, runtime = log_line.strip("\n").split(key_phrase)
            hours, minutes, seconds = map(int, runtime.split(":"))
            as_seconds = (60 * 60 * hours) + (60 * minutes) + seconds
            jobs[job_id] = as_seconds

# Convert the sum of seconds to hours, minutes, seconds
seconds_sum = sum([jobs[job_id] for job_id in jobs.keys()])
print "Total seconds of runtime: %d" % seconds_sum

total_days = seconds_sum / (24 * 60 * 60)   # Implicitly floors
total_hours = (seconds_sum - (24 * 60 * 60 * total_days)) / (60 * 60)   # Implicitly floors
total_minutes = (seconds_sum - (24 * 60 * 60 * total_days) - (60 * 60 * total_hours)) / 60    # Implicitly floors
total_seconds = seconds_sum - (24 * 60 * 60 * total_days) - (60 * 60 * total_hours) - (60 * total_minutes)

print "Total runtime used: %d days, %d hours, %d minutes, %d seconds" % (total_days, total_hours, total_minutes, total_seconds)