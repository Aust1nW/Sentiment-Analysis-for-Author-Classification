from keras import models
from keras import layers
from keras.preprocessing.sequence import pad_sequences
import random
import json
import re
import numpy as np


def build_model():
    vocab_size = 18708
    max_len = 23
    nn = models.Sequential()
    nn.add(layers.Embedding(vocab_size, 16, input_length=max_len))
    nn.add(layers.LSTM(32))
    nn.add(layers.Dense(1, activation='sigmoid'))
    return nn


def read_data(file_name):
    data = []
    with open(file_name, 'r') as file:
        for file_line in file:
            file_line = re.sub(r',(?=(((?!\]).)*\[)|[^\[\]]*$)|\n',
                               '', file_line)
            data.append(json.loads(file_line))
    return data


def split_data(data):
    cut_off = round(len(data) * .9)
    train_data = data[:cut_off]
    val_data = data[cut_off+1:]
    return train_data, val_data


def avg_length(data):
    count = 0
    for x in data:
        count += len(x[0])
    return round(count/len(data))


def generator(a_data, s_data, batch):
    # TODO Finish generator
    a_iterator = 0
    s_iterator = 0
    mean = 23
    while 1:
        sample = []
        label = []
        rand = random.randint(0, 1)
        for i in range(batch):
            if rand:
                sample.append(a_data[a_iterator])
                label.append([1])
                a_iterator = (a_iterator + 1) % len(a_data)
            else:
                sample.append(s_data[s_iterator])
                label.append([0])
                s_iterator = (s_iterator + 1) % len(s_data)
        output = pad_sequences(sample, maxlen=mean)
        yield output, np.array(label)


# Read in the data and split it between training and validation
austen_data = read_data('Austen.train')
stoker_data = read_data('Stoker.train')

# Split the data to training and validation data
a_train_data, a_val_data = split_data(austen_data)
s_train_data, s_val_data = split_data(stoker_data)

# Generators for train validation and test data
data_gen = generator(a_train_data, s_train_data, 128)
val_gen = generator(a_val_data, s_val_data, 128)


# Build the model and run
steps_epoch = 128
epochs = 5
val_steps = 128
model = build_model()
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit_generator(data_gen,
                              steps_per_epoch=steps_epoch,
                              epochs=epochs,
                              validation_data=val_gen,
                              validation_steps=val_steps)
model.save('PorV.h5')

# TODO Create Embedding.dat
embed = open("Embedding.dat", 'w')
embed.write('\n')
embedding = model.layers[0].get_weights()[0]
embed_string = ''
for line in embedding:
    line = re.sub(r'\[', ' ', str(line))
    line = re.sub('\n', ' ', line)
    line = re.sub(r']', ',', line)
    embed_string += line
embed_arr = embed_string.split(',')
for line in embed_arr:
    arr = []
    li = line.split(' ')
    for i in li:
        if i.isspace() or len(i) == 0:
            continue
        else:
            arr.append(float(i))

    if len(arr) == 0:
        break
    embed.write(str(arr))
    embed.write('\n')
