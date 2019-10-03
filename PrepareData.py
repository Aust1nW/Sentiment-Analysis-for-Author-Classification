import sys
from keras.preprocessing.text import Tokenizer
import re


def parse_the_file(file):

    str_tmp = ''
    flag = 0
    for line in file:
        # If the line consists of only a new line, then move on
        if '\n' in line and len(line) == 1:
            continue

        # Ignores chapter headings
        if flag:
            flag -= flag
            continue
        if 'CHAPTER' in line:
            flag = 2
            continue

        # Remove everything in between ### and --
        line = re.sub(r'\###[^]]*\--', '', line)
        # Remove everything between ( )
        line = re.sub(r'\([^]]*\)', '', line)
        # Remove everything between ( and _
        line = re.sub(r'\([^]]*\_', '', line)
        line = re.sub('Mr.', '', line)
        line = re.sub('Mrs.', '', line)
        # Remove :--
        line = re.sub(':--', '', line)
        if len(line) > 1:
            str_tmp += line

    # Split the text up into sentences
    tensor = str_tmp.split('.')
    result = []
    for line in tensor:
        line = re.sub(r'\'', ' ', line)
        line = re.sub(r'[^\w\s]', ' ', line)
        line = re.sub('\n', '', line)
        line = re.sub(r'[\d-]', '', line)
        line = line.lstrip()
        line = line.rstrip()
        line = line.lower()
        if len(line) > 0:
            result.append(line)
    return result


def build_json(book):
    list = []
    for sequence in book:
        list.append(sequence)
    return list


def write_outputs(train, test, sequences):
    train_length = round(len(sequences)*.75)
    for i in range(0, train_length):
        train.write(str(sequences[i]))
        train.write('\n')

    for j in range(train_length, len(sequences)):
        test.write(str(sequences[j]))
        test.write('\n')


def generate_vocab(vocab, word_index):
    # TODO Finish Vocab.dat
    vocab.write('\n')
    for i in word_index:
        vocab.write(i + '\n')


# Open each file
austen = open(sys.argv[1], "r")
stoker = open(sys.argv[2], "r")

# Parse each file and return a list of lists of sentences
dracula = parse_the_file(stoker)
pAndP = parse_the_file(austen)

max_sequences = 18975

# Declare the tokenizer
tokenizer = Tokenizer(num_words=max_sequences)
tokenizer.fit_on_texts(dracula + pAndP)

# Text to sequences for both inputs
dracula_sequences = tokenizer.texts_to_sequences(dracula)
p_sequences = tokenizer.texts_to_sequences(pAndP)

# Get the word index
word_index = tokenizer.word_index

# Write Vocab.dat
vocab = open('Vocab.dat', 'w')
generate_vocab(vocab, word_index)


# Build json sequences
dracula_json = build_json(dracula_sequences)
p_json = build_json(p_sequences)

# Open each train/test file and start writing
austen_train = open('Austen.train', 'w')
austin_test = open('Austen.test', 'w')
write_outputs(austen_train, austin_test, p_json)

stoker_train = open('Stoker.train', 'w')
stoker_test = open('Stoker.test', 'w')
write_outputs(stoker_train, stoker_test, dracula_json)
