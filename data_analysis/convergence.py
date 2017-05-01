import matplotlib.pyplot as plt
import numpy as np

import sys

if len(sys.argv) != 2:
    print "Usage: python convergence.py <log (Slurm) file>"
    sys.exit(1)

log_filename = sys.argv[1]

# Read various accuracies and plot them
training_fers = []
validation_fers = []
validation_uers = []
reading_training = False
early_stop = False
with open(log_filename, 'r') as log:
    for log_line in log.readlines():
        if not reading_training:
            training_phrase = "Training confusion matrix:"
            validation_fer_phrase = "  FER: "
            validation_uer_phrase = "  UER: "
            early_stop_phrase = "STOPPING EARLY"

            if training_phrase in log_line:
                # We should expect a training Frame Error Rate within a few lines
                # (after the ConfusionMatrix)
                reading_training = True
            elif validation_fer_phrase in log_line:
                # Found a validation Frame Error Rate
                validation_fer = float(log_line.split(validation_fer_phrase)[1].strip("%\n"))
                validation_fers.append(validation_fer)
            elif validation_uer_phrase in log_line:
                # Found a validation Utterance Error Rate
                validation_uer = float(log_line.split(validation_uer_phrase)[1].strip("%\n"))
                validation_uers.append(validation_uer)
            elif early_stop_phrase in log_line:
                # Early stop - cut off last error rates from output later
                early_stop = True
        else:
            key_phrase = " + global correct: "
            if key_phrase in log_line:
                training_fer = float(log_line.split(key_phrase)[1].strip("%\n"))
                training_fers.append(training_fer)
                reading_training = False

# Make sure we actually read the data correctly
assert(len(training_fers) == len(validation_fers))
assert(len(validation_fers) == len(validation_uers))

# Cut off last data points if we stopped early, as this model was not used
if early_stop:
    training_fers = training_fers[:-1]
    validation_fers = validation_fers[:-1]
    validation_uers = validation_uers[:-1]

# Convert validation error rates to percentages (training FER is already percentage)
validation_fers = map(lambda x: 100.0 * x, validation_fers)
validation_uers = map(lambda x: 100.0 * x, validation_uers)

# Convert training accuracy to FER
training_fers = map(lambda x: 100.0 - x, training_fers)

# Plot it!
epochs = [x + 1 for x in xrange(len(training_fers))]
plt.plot(epochs, training_fers, 'r', label="Training FER")
plt.plot(epochs, validation_fers, 'g', label="Validation FER")
plt.plot(epochs, validation_uers, 'b', label="Validation UER")
plt.ylabel("Error Rate (%)")
plt.xlabel("Epoch")
plt.legend(loc="upper right", shadow=True)
plt.axis([1, len(epochs), 0.0, 100.0])
plt.xticks(np.arange(1, len(epochs) + 1, 2))
plt.yticks(np.arange(0.0, 100.0, 10.0))
plt.show()
