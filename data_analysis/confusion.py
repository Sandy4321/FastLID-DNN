import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
import sys

if len(sys.argv) != 3:
    print "Usage: python confusion.py <log (Slurm) file> <# classes>"
    sys.exit(1)

log_filename = sys.argv[1]
num_classes = int(sys.argv[2])

# Read in last confusion matrix and plot it
count2labels = {2: ["out", "in"],
                3: ["english", "german", "mandarin"],
                4: ["oos", "english", "german", "mandarin"]}
labels = count2labels[num_classes]
conf_mat = np.zeros((num_classes, num_classes), np.float64)
current_row = -1    # Indicates that we are not currently reading
with open(log_filename, 'r') as log:
    for log_line in log.readlines():
        if current_row < 0:
            # Check if we should start reading a confusion matrix
            key_phrase = "ConfusionMatrix"
            if key_phrase in log_line:
                current_row = 0
        else:
            key_phrase = "["
            if key_phrase not in log_line:
                current_row = -1
            else:
                stripped_line = log_line.split("[  ")[1].split("]")[0]
                data = map(float, filter(lambda x: len(x) > 0, stripped_line.split(" ")))
                assert(len(data) == num_classes)
                conf_mat[current_row] = data
                current_row += 1

# Normalize confusion matrix and convert to percentages
for i in xrange(num_classes):
    row_sum = sum(conf_mat[i])
    conf_mat[i] = np.multiply(conf_mat[i], 100.0 / row_sum)

# Confusion matrix plotting with help of
# http://stackoverflow.com/questions/35572000/how-can-i-plot-a-confusion-matrix
df_cm = pd.DataFrame(conf_mat, index = labels, columns = labels)
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True, vmin=0.0, vmax=100.0)
plt.show()
