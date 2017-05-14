import numpy as np
import os
import sys

if len(sys.argv) != 2:
    print "Usage: python misclassified.py <log file directory>"
    sys.exit(1)

log_file_directory = sys.argv[1]

# Read in misclassified utterances for each network
misclassified = dict()  # Maps network name to list of (utterance ID, label, classification) tuples
all_utts = dict()       # Maps utterance ID to networks that misclassified it
for eval_filename in filter(lambda x: "evaluate_" in x, os.listdir(log_file_directory)):
    net_name = eval_filename.rstrip(".log").split("evaluate_")[1]
    misclassified_utts = []
    with open("%s/%s" % (log_file_directory, eval_filename), 'r') as log:
        key_phrase = "Incorrectly classified "
        for log_line in log.readlines():
            if key_phrase in log_line:
                raw_info = log_line.strip("\n").split(key_phrase)[1]

                label_split = raw_info.split(" utterance ")
                label = label_split[0]

                utt_split = label_split[1].split(" as ")
                utt_id = utt_split[0]
                classification = utt_split[1].strip("\t")

                full_info = (utt_id, label, classification)
                misclassified_utts.append(full_info)

                if utt_id in all_utts.keys():
                    all_utts[utt_id].append(net_name)
                else:
                    all_utts[utt_id] = [net_name]
    misclassified[net_name] = misclassified_utts

for net_name in sorted(misclassified.keys()):
    print "Network %s misclassifications:" % net_name
    print misclassified[net_name]

for utt_id in sorted(all_utts.keys()):
    # print "Utterance %s misclassified by %d networks: %s" % (utt_id, len(all_utts[utt_id]), all_utts[utt_id])
    if len(all_utts[utt_id]) == 5:
        print "Utterance %s misclassified by all 5 networks" % utt_id

