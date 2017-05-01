import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import sys

if len(sys.argv) != 2:
    print "Usage: python early_stopping.py <log (Slurm) file>"
    sys.exit(1)

log_filename = sys.argv[1]

# Read in validation FER and generalization losses and plot the lines compared to each other
validation_fers = []
generalization_losses = []
reading_gl = False
with open(log_filename, 'r') as log:
    for log_line in log.readlines():
        if not reading_gl:
            # Check if we should start reading a validation FER
            fer_phrase = "FER: "
            gl_phrase = "Generalization loss:"
            if fer_phrase in log_line:
                fer = float(log_line.split(fer_phrase)[1].strip())
                validation_fers.append(fer)
            elif gl_phrase in log_line:
                reading_gl = True
        else:
            # Read generalization loss from this following line
            generalization_loss = float(log_line.strip())
            generalization_losses.append(generalization_loss)
            reading_gl = False

assert(len(validation_fers) == len(generalization_losses))

# Convert FER decimals to percentages
validation_fers = map(lambda x: 100.0 * x, validation_fers)

gl_threshold = 10.0

fig, ax1 = plt.subplots()
ax1.plot(range(len(validation_fers)), validation_fers, 'b')
ax2 = ax1.twinx()
ax2.plot(range(len(generalization_losses)), generalization_losses, 'r',
         range(len(generalization_losses)), gl_threshold * np.ones(len(generalization_losses)), 'g')
fig.tight_layout()
plt.show()
