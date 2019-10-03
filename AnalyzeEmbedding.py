import json
import re
import sys
import numpy as np


def read_data(file_name):
    data = []
    c = 0
    data.append([])
    with open(file_name, 'r') as file:
        for file_line in file:
            file_line = re.sub(r',(?=(((?!\]).)*\[)|[^\[\]]*$)|\n',
                               '', file_line)
            if c > 0:
                data.append(json.loads(file_line))
            else:
                c = c + 1
    return np.array(data)


def search_for_top_ten(matrix, dim):
    tmp = []
    # Get all the values for the dimension associated with the row number
    for j in range(1, 18708):
        a = []
        a.append(j)
        a.append(matrix[j][dim])
        tmp.append(a)
    hi = sorted(tmp, key=lambda x: x[1], reverse=True)
    lo = sorted(tmp, key=lambda x: x[1])
    return hi[:10], lo[:10]


def print_dimensions(h, l, v, dim):
    print("Dimension " + str(dim+1) + " : ")
    print("Top 10: ", end='')
    for li in range(10):
        vocab_index = h[li][0]
        print(str(v[vocab_index]) + ' ', end='')
    print('')
    print("Lowest 10: ", end='')
    for li in range(10):
        vocab_index = l[li][0]
        if vocab_index > len(v):
            print(vocab_index)
        print(str(v[vocab_index]), end='')
    print('\n')


embedding = read_data(sys.argv[2])
vocab_file = open(sys.argv[1], 'r')
vocab = []
vocab.append('')
for line in vocab_file:
    line = line.rsplit()
    vocab.append(line)

count = 0
for i in range(0, 16):
    high, low = search_for_top_ten(embedding, i)
    print_dimensions(high, low, vocab, i)
