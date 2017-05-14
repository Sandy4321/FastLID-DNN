import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter
from scipy.stats import mode
import sys

if len(sys.argv) != 2:
    print "Usage: python posteriors.py <log (Slurm) file>"
    sys.exit(1)

log_filename = sys.argv[1]

# Read in frame-level posteriors for each utterance
print "Reading in frame-level posteriors..."
log_posteriors = []     # List of (utterance id, label, array of posterior log probabilities per frame) tuples
log_idx = -1    # Set as such because it will be increased before being used
with open(log_filename, 'r') as log:
    for log_line in log.readlines():
        new_utt_phrase = " WITH LABEL "
        posterior_phrase = "posteriors:"
        if new_utt_phrase in log_line:
            # We have a new utterance!
            raw_data = log_line.lstrip("UTTERANCE ").strip("\n").split(new_utt_phrase)
            utt_id = raw_data[0]
            label = int(raw_data[1])
            log_idx += 1
            log_posteriors.append((utt_id, label, []))
        elif posterior_phrase in log_line:
            # Add posteriors for current utterances
            raw_data = log_line.strip("\n").split(posterior_phrase)[1]
            formatted_data = map(float, raw_data.split(","))
            log_posteriors[log_idx][2].append(formatted_data)

total_utterances = len(log_posteriors)
print "Read posteriors for %d utterances" % total_utterances

print "Classifying based on average probabilities..."
correct = 0
for i in xrange(total_utterances):
    label = log_posteriors[i][1]
    log_posteriors_matrix = np.asarray(log_posteriors[i][2])
    num_posteriors = log_posteriors_matrix.shape[0]
    avg_posteriors = np.divide(np.sum(log_posteriors_matrix, axis=0),
                               np.ones(log_posteriors_matrix.shape[1]) * num_posteriors)
    classification = np.argmax(avg_posteriors) + 1      # The classifier used 1-indexing...
    if classification == label:
        correct += 1
print "\tClassified with UER %.3f" % (1.0 - (correct / float(total_utterances)))

print "Classifying based on max probabilities..."
correct = 0
for i in xrange(total_utterances):
    label = log_posteriors[i][1]
    log_posteriors_matrix = np.asarray(log_posteriors[i][2])
    classification = (np.argmax(log_posteriors_matrix) % 4) + 1     # The classifier used 1-indexing...
    if classification == label:
        correct += 1
print "\tClassified with UER %.3f" % (1.0 - (correct / float(total_utterances)))

print "Classifying based on probability modes..."
correct = 0
for i in xrange(total_utterances):
    label = log_posteriors[i][1]
    log_posteriors_matrix = np.asarray(log_posteriors[i][2])
    num_posteriors = log_posteriors_matrix.shape[0]
    frame_classifications = [np.argmax(log_posteriors_matrix[i, :]) for i in xrange(num_posteriors)]
    classification = mode(frame_classifications)[0] + 1     # The classifier used 1-indexing...
    if classification == label:
        correct += 1
print "\tClassified with UER %.3f" % (1.0 - (correct / float(total_utterances)))

print "Classifying based on longest run..."
correct = 0
for i in xrange(total_utterances):
    label = log_posteriors[i][1]
    log_posteriors_matrix = np.asarray(log_posteriors[i][2])
    num_posteriors = log_posteriors_matrix.shape[0]

    longest_run = 0
    longest_classification = -1     # Forces us to evaluate first element
    current_run = 0
    current_classification = -1     # Forces us to evaluate first element
    for j in xrange(num_posteriors):
        frame_classification = np.argmax(log_posteriors_matrix[j, :])
        if frame_classification != current_classification:
            current_classification = frame_classification
            current_run = 0     # Reset --- it will be incremented below

        current_run += 1
        if current_run > longest_run:
            longest_classification = current_classification
            longest_run += 1

    classification = longest_classification + 1     # The classifier used 1-indexing...
    if classification == label:
        correct += 1
print "\tClassified with UER %.3f" % (1.0 - (correct / float(total_utterances)))

lpf_order = 3
lpf_normfreq = 0.05
lpf_b, lpf_a = butter(lpf_order, lpf_normfreq)
print "Classifying based on average of low-pass filtered posteriors..."
correct = 0
for i in xrange(total_utterances):
    label = log_posteriors[i][1]
    log_posteriors_matrix = np.asarray(log_posteriors[i][2])
    filtered_posteriors_matrix = np.asarray([lfilter(lpf_b, lpf_a, log_posteriors_matrix[:, j]) for j in xrange(log_posteriors_matrix.shape[1])]).T
    num_posteriors = filtered_posteriors_matrix.shape[0]
    avg_posteriors = np.divide(np.sum(filtered_posteriors_matrix, axis=0),
                               np.ones(filtered_posteriors_matrix.shape[1]) * num_posteriors)
    classification = np.argmax(avg_posteriors) + 1      # The classifier used 1-indexing...
    if classification == label:
        correct += 1
print "\tClassified with UER %.3f" % (1.0 - (correct / float(total_utterances)))



# Normal moving average

window = 5
print "Moving average (window = %d)..." % window
smoothed_posteriors = []
for i in xrange(total_utterances):
    log_posteriors_matrix = np.asarray(log_posteriors[i][2])
    num_classes = log_posteriors_matrix.shape[1]
    smoothed = []
    for i in xrange(num_classes):
        # Avoid the NaN at the beginning
        smoothed.append(pd.Series(log_posteriors_matrix[:, i]).rolling(window=window).mean()[window - 1:])
    smoothed = np.asarray(smoothed).T
    smoothed_posteriors.append(smoothed)
smoothed_posteriors = np.asarray(smoothed_posteriors)

print "\tClassifying based on average probabilities..."
correct = 0
for i in xrange(total_utterances):
    label = log_posteriors[i][1]
    smoothed_posteriors_matrix = np.asarray(smoothed_posteriors[i])
    num_posteriors = smoothed_posteriors_matrix.shape[0]
    avg_posteriors = np.divide(np.sum(smoothed_posteriors_matrix, axis=0),
                               np.ones(smoothed_posteriors_matrix.shape[1]) * num_posteriors)
    classification = np.argmax(avg_posteriors) + 1      # The classifier used 1-indexing...
    if classification == label:
        correct += 1
print "\t\tClassified with UER %.3f" % (1.0 - (correct / float(total_utterances)))

print "\tClassifying based on max probabilities..."
correct = 0
for i in xrange(total_utterances):
    label = log_posteriors[i][1]
    smoothed_posteriors_matrix = np.asarray(smoothed_posteriors[i])
    classification = (np.argmax(smoothed_posteriors_matrix) % 4) + 1     # The classifier used 1-indexing...
    if classification == label:
        correct += 1
print "\t\tClassified with UER %.3f" % (1.0 - (correct / float(total_utterances)))

print "\tClassifying based on probability modes..."
correct = 0
for i in xrange(total_utterances):
    label = log_posteriors[i][1]
    smoothed_posteriors_matrix = np.asarray(smoothed_posteriors[i])
    num_posteriors = smoothed_posteriors_matrix.shape[0]
    frame_classifications = [np.argmax(smoothed_posteriors_matrix[i, :]) for i in xrange(num_posteriors)]
    classification = mode(frame_classifications)[0] + 1     # The classifier used 1-indexing...
    if classification == label:
        correct += 1
print "\t\tClassified with UER %.3f" % (1.0 - (correct / float(total_utterances)))

print "\tClassifying based on longest run..."
correct = 0
for i in xrange(total_utterances):
    label = log_posteriors[i][1]
    smoothed_posteriors_matrix = np.asarray(smoothed_posteriors[i])
    num_posteriors = smoothed_posteriors_matrix.shape[0]

    longest_run = 0
    longest_classification = -1     # Forces us to evaluate first element
    current_run = 0
    current_classification = -1     # Forces us to evaluate first element
    for j in xrange(num_posteriors):
        frame_classification = np.argmax(smoothed_posteriors_matrix[j, :])
        if frame_classification != current_classification:
            current_classification = frame_classification
            current_run = 0     # Reset --- it will be incremented below

        current_run += 1
        if current_run > longest_run:
            longest_classification = current_classification
            longest_run += 1

    classification = longest_classification + 1     # The classifier used 1-indexing...
    if classification == label:
        correct += 1
print "\t\tClassified with UER %.3f" % (1.0 - (correct / float(total_utterances)))

lpf_order = 3
lpf_normfreq = 0.05
lpf_b, lpf_a = butter(lpf_order, lpf_normfreq)
print "\tClassifying based on average of low-pass filtered posteriors..."
correct = 0
for i in xrange(total_utterances):
    label = log_posteriors[i][1]
    smoothed_posteriors_matrix = np.asarray(smoothed_posteriors[i])
    filtered_posteriors_matrix = np.asarray([lfilter(lpf_b, lpf_a, smoothed_posteriors_matrix[:, j]) for j in xrange(smoothed_posteriors_matrix.shape[1])]).T
    num_posteriors = filtered_posteriors_matrix.shape[0]
    avg_posteriors = np.divide(np.sum(filtered_posteriors_matrix, axis=0),
                               np.ones(filtered_posteriors_matrix.shape[1]) * num_posteriors)
    classification = np.argmax(avg_posteriors) + 1      # The classifier used 1-indexing...
    if classification == label:
        correct += 1
print "\t\tClassified with UER %.3f" % (1.0 - (correct / float(total_utterances)))



# Exponentially weighted moving average

window = 10
print "Exponentially weight moving average (window = %d)..." % window
smoothed_posteriors = []
for i in xrange(total_utterances):
    log_posteriors_matrix = np.asarray(log_posteriors[i][2])
    num_classes = log_posteriors_matrix.shape[1]
    smoothed = []
    for i in xrange(num_classes):
        # Avoid the NaN at the beginning
        smoothed.append(pd.Series(log_posteriors_matrix[:, i]).ewm(span=window).mean()[window - 1:])
    smoothed = np.asarray(smoothed).T
    smoothed_posteriors.append(smoothed)
smoothed_posteriors = np.asarray(smoothed_posteriors)

print "\tClassifying based on average probabilities..."
correct = 0
for i in xrange(total_utterances):
    label = log_posteriors[i][1]
    smoothed_posteriors_matrix = np.asarray(smoothed_posteriors[i])
    num_posteriors = smoothed_posteriors_matrix.shape[0]
    avg_posteriors = np.divide(np.sum(smoothed_posteriors_matrix, axis=0),
                               np.ones(smoothed_posteriors_matrix.shape[1]) * num_posteriors)
    classification = np.argmax(avg_posteriors) + 1      # The classifier used 1-indexing...
    if classification == label:
        correct += 1
print "\t\tClassified with UER %.3f" % (1.0 - (correct / float(total_utterances)))

print "\tClassifying based on max probabilities..."
correct = 0
for i in xrange(total_utterances):
    label = log_posteriors[i][1]
    smoothed_posteriors_matrix = np.asarray(smoothed_posteriors[i])
    classification = (np.argmax(smoothed_posteriors_matrix) % 4) + 1     # The classifier used 1-indexing...
    if classification == label:
        correct += 1
print "\t\tClassified with UER %.3f" % (1.0 - (correct / float(total_utterances)))

print "\tClassifying based on probability modes..."
correct = 0
for i in xrange(total_utterances):
    label = log_posteriors[i][1]
    smoothed_posteriors_matrix = np.asarray(smoothed_posteriors[i])
    num_posteriors = smoothed_posteriors_matrix.shape[0]
    frame_classifications = [np.argmax(smoothed_posteriors_matrix[i, :]) for i in xrange(num_posteriors)]
    classification = mode(frame_classifications)[0] + 1     # The classifier used 1-indexing...
    if classification == label:
        correct += 1
print "\t\tClassified with UER %.3f" % (1.0 - (correct / float(total_utterances)))

print "\tClassifying based on longest run..."
correct = 0
for i in xrange(total_utterances):
    label = log_posteriors[i][1]
    smoothed_posteriors_matrix = np.asarray(smoothed_posteriors[i])
    num_posteriors = smoothed_posteriors_matrix.shape[0]

    longest_run = 0
    longest_classification = -1     # Forces us to evaluate first element
    current_run = 0
    current_classification = -1     # Forces us to evaluate first element
    for j in xrange(num_posteriors):
        frame_classification = np.argmax(smoothed_posteriors_matrix[j, :])
        if frame_classification != current_classification:
            current_classification = frame_classification
            current_run = 0     # Reset --- it will be incremented below

        current_run += 1
        if current_run > longest_run:
            longest_classification = current_classification
            longest_run += 1

    classification = longest_classification + 1     # The classifier used 1-indexing...
    if classification == label:
        correct += 1
print "\t\tClassified with UER %.3f" % (1.0 - (correct / float(total_utterances)))

lpf_order = 3
lpf_normfreq = 0.05
lpf_b, lpf_a = butter(lpf_order, lpf_normfreq)
print "\tClassifying based on average of low-pass filtered posteriors..."
correct = 0
for i in xrange(total_utterances):
    label = log_posteriors[i][1]
    smoothed_posteriors_matrix = np.asarray(smoothed_posteriors[i])
    filtered_posteriors_matrix = np.asarray([lfilter(lpf_b, lpf_a, smoothed_posteriors_matrix[:, j]) for j in xrange(smoothed_posteriors_matrix.shape[1])]).T
    num_posteriors = filtered_posteriors_matrix.shape[0]
    avg_posteriors = np.divide(np.sum(filtered_posteriors_matrix, axis=0),
                               np.ones(filtered_posteriors_matrix.shape[1]) * num_posteriors)
    classification = np.argmax(avg_posteriors) + 1      # The classifier used 1-indexing...
    if classification == label:
        correct += 1
print "\t\tClassified with UER %.3f" % (1.0 - (correct / float(total_utterances)))

