import matplotlib.pyplot as plt

import sys

if len(sys.argv) != 2:
    print "Usage: python convergence.py <log (Slurm) file>"
    sys.exit(1)

log_filename = sys.argv[1]

# Read accuracies and plot them
epoch_accuracies = []
with open(log_filename, 'r') as log:
    for log_line in log.readlines():
        key_phrase = " + global correct: "
        if key_phrase in log_line:
            epoch_accuracy = float(log_line.split(key_phrase)[1].strip("%\n"))
            epoch_accuracies.append(epoch_accuracy)

plt.plot([x + 1 for x in xrange(len(epoch_accuracies))], epoch_accuracies)
plt.ylabel("Classification Accuracy (%)")
plt.xlabel("Epoch")
plt.axis([1, len(epoch_accuracies), 0.0, 100.0])
plt.show()