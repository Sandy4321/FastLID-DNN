import matplotlib.pyplot as plt
import numpy as np
import sys
import wave

if len(sys.argv) != 3:
    print "Usage: python posterior.py <log (Slurm) file> <wav file>"
    sys.exit(1)

log_filename = sys.argv[1]
wav_filename = sys.argv[2]

# Read in audio file as array
utt_wav = wave.open(wav_filename,'r')
signal = utt_wav.readframes(-1)
signal = np.fromstring(signal, 'Int16')

# Read in frame-level posteriors and plot the lines compared to each other
log_posteriors = []
with open(log_filename, 'r') as log:
    for log_line in log.readlines():
        key_phrase = "posteriors:"
        if key_phrase in log_line:
            raw_data = log_line.strip("\n").split(key_phrase)[1]
            processed_data = map(float, raw_data.split(","))
            log_posteriors.append(processed_data)

# Convert to regular posteriors
posteriors = np.exp(log_posteriors)

'''
# Threshold the posteriors to only high-confidence ones
threshold = 0.95
def posterior_threshold(pvec):
    if max(pvec) >= threshold:
        return pvec
    else:
        return np.zeros(len(pvec))
thresholded_posteriors = np.asarray([posterior_threshold(posteriors[row, :]) for row in xrange(posteriors.shape[0])])
'''

# Plot on separate subplots
f, axes = plt.subplots(2, 1)
axes[0].plot(signal)
axes[0].set_ylabel("Audio Amplitude (16-bit quantized)")

oos_line = axes[1].plot(range(posteriors.shape[0]), posteriors[:, 0], label="Out-of-Set")
english_line = axes[1].plot(range(posteriors.shape[0]), posteriors[:, 1], label="English")
german_line = axes[1].plot(range(posteriors.shape[0]), posteriors[:, 2], label="German")
mandarin_line = axes[1].plot(range(posteriors.shape[0]), posteriors[:, 3], label="Mandarin")
axes[1].set_ylabel("Posterior Probability")

'''
oos_line = axes[1].plot(range(thresholded_posteriors.shape[0]), thresholded_posteriors[:, 0], label="Out-of-Set")
english_line = axes[1].plot(range(thresholded_posteriors.shape[0]), thresholded_posteriors[:, 1], label="English")
german_line = axes[1].plot(range(thresholded_posteriors.shape[0]), thresholded_posteriors[:, 2], label="German")
mandarin_line = axes[1].plot(range(thresholded_posteriors.shape[0]), thresholded_posteriors[:, 3], label="Mandarin")
axes[1].set_ylabel("Thresholded Posterior Probability")
'''

lines = oos_line + english_line + german_line + mandarin_line
labels = [line.get_label() for line in lines]
axes[1].legend(lines, labels)

plt.show()
