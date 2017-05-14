import matplotlib.pyplot as plt
import numpy as np
import sys
import wave

if len(sys.argv) != 3:
    print "Usage: python posterior_viz.py <log (Slurm) file> <utt ID>"
    sys.exit(1)

log_filename = sys.argv[1]
utt_id = sys.argv[2]
wav_filename = "%s.wav" % utt_id

# Read in audio file as array
utt_wav = wave.open(wav_filename,'r')
signal = utt_wav.readframes(-1)
signal = np.fromstring(signal, 'Int16')

# Read in frame-level posteriors and plot the lines compared to each other
log_posteriors = []
reading_posteriors = False
with open(log_filename, 'r') as log:
    for log_line in log.readlines():
        if not reading_posteriors:
            utt_phrase = "%s WITH LABEL " % utt_id
            if utt_phrase in log_line:
                # We have our utterance!
                reading_posteriors = True
        else:
            other_utt_phrase = " WITH LABEL "
            posterior_phrase = "posteriors:"
            if other_utt_phrase in log_line:
                # We're done yo
                reading_posteriors = False
            elif posterior_phrase in log_line:
                # Add posteriors for this frame
                raw_data = log_line.strip("\n").split(posterior_phrase)[1]
                processed_data = map(float, raw_data.split(","))
                log_posteriors.append(processed_data)

# Convert to regular posteriors
posteriors = np.exp(log_posteriors)

# Plot on separate subplots
f, axes = plt.subplots(2, 1)
axes[0].plot(signal)
axes[0].set_ylabel("Audio Amplitude (16-bit quantized)")

oos_line = axes[1].plot(range(posteriors.shape[0]), posteriors[:, 0], label="Out-of-Set")
english_line = axes[1].plot(range(posteriors.shape[0]), posteriors[:, 1], label="English")
german_line = axes[1].plot(range(posteriors.shape[0]), posteriors[:, 2], label="German")
mandarin_line = axes[1].plot(range(posteriors.shape[0]), posteriors[:, 3], label="Mandarin")
axes[1].set_ylabel("Posterior Probability")

lines = oos_line + english_line + german_line + mandarin_line
labels = [line.get_label() for line in lines]
axes[1].legend(lines, labels)

plt.show()
