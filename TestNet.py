from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import json
import re
import sys


# Read in the test data
mean = 23
test_data = []

for line in sys.stdin:
    line = re.sub(r',(?=(((?!\]).)*\[)|[^\[\]]*$)|\n', '', line)
    test_data.append(json.loads(line))
test = pad_sequences(test_data, maxlen=mean)

austen = 0
stoker = 0

# Load the model from input h5
nn = load_model(sys.argv[1])
output_labels = nn.predict(test)
for i in output_labels:
    if i > 0.5:
        austen = austen + 1
    else:
        stoker = stoker + 1

austen_percentage = austen/len(test)
stoker_percentage = stoker/len(test)
print("\nAusten : " + str(round(austen_percentage, 4)) + "\n")
print("Stoker : " + str(round(stoker_percentage, 4)) + "\n")

# TODO Figure out how to calculate percentages for result
