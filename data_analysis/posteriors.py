import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, lfilter
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
print "Classified with UER %.3f" % (1.0 - (correct / float(total_utterances)))

print "Classifying based on max probabilities..."
correct = 0
for i in xrange(total_utterances):
    label = log_posteriors[i][1]
    log_posteriors_matrix = np.asarray(log_posteriors[i][2])
    classification = (np.argmax(log_posteriors_matrix) % 4) + 1     # The classifier used 1-indexing...
    if classification == label:
        correct += 1
print "Classified with UER %.3f" % (1.0 - (correct / float(total_utterances)))

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
print "Classified with UER %.3f" % (1.0 - (correct / float(total_utterances)))

'''
f, axes = plt.subplots(4, 1)
axes[0].set_ylabel("Log Posterior Probability")
axes[1].set_ylabel("Log Posterior Probability")
axes[2].set_ylabel("Log Posterior Probability")
axes[3].set_ylabel("Log Posterior Probability")

unfiltered = axes[0].plot(range(num_posteriors), log_posteriors_matrix[:, 0], label="Out-of-Set (u)")
filtered = axes[0].plot(range(num_posteriors), filtered_posteriors_matrix[:, 0], label="Out-of-Set (f)")
lines = unfiltered + filtered
labels = [line.get_label() for line in lines]
axes[0].legend(lines, labels)

unfiltered = axes[1].plot(range(num_posteriors), log_posteriors_matrix[:, 1], label="English (u)")
filtered = axes[1].plot(range(num_posteriors), filtered_posteriors_matrix[:, 1], label="English (f)")
lines = unfiltered + filtered
labels = [line.get_label() for line in lines]
axes[1].legend(lines, labels)

unfiltered = axes[2].plot(range(num_posteriors), log_posteriors_matrix[:, 2], label="German (u)")
filtered = axes[2].plot(range(num_posteriors), filtered_posteriors_matrix[:, 2], label="German (f)")
lines = unfiltered + filtered
labels = [line.get_label() for line in lines]
axes[2].legend(lines, labels)

unfiltered = axes[3].plot(range(num_posteriors), log_posteriors_matrix[:, 3], label="Mandarin (u)")
filtered = axes[3].plot(range(num_posteriors), filtered_posteriors_matrix[:, 3], label="Mandarin (f)")
lines = unfiltered + filtered
labels = [line.get_label() for line in lines]
axes[3].legend(lines, labels)

plt.show()
'''
