import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import LSTM, RepeatVector, Embedding
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import SGD
import numpy as np
import pickle
import math
import time
import two_sum_question as tsq

pickle_in = open("./model-development/X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("./model-development/y.pickle", "rb")
y = pickle.load(pickle_in)

SAMPLES = 2000000
BATCH_SIZE = 128
HIDDEN_LAYERS = 3
EPOCHS = 20
NEURONS = 256

SMALLEST_SAFE_NUMBER = 255  # 11111111
LEN_NUMS = len(tsq.nums)

largest_number = max(max(tsq.nums), tsq.target)
largest_number = SMALLEST_SAFE_NUMBER if SMALLEST_SAFE_NUMBER > largest_number else largest_number + 1
digits = len("{0:b}".format(largest_number))
largest_bin_number = int('1'*digits, 2)
upper_bound = math.floor(largest_bin_number/2)


def get_num_classes():
    num_classes = 0
    for i in range(LEN_NUMS - 1, 0, -1):
        num_classes += i
    return num_classes


# get the number of classes
num_classes = get_num_classes()


def bin_encode(x):
    binaryStr = "{0:b}".format(x)
    if len(binaryStr) != digits:
        num_zeroes = digits - len(binaryStr)
        binaryStr = '0'*num_zeroes + binaryStr
    b_digits = []
    for b_str_digit in binaryStr:
        b_digits.append(int(b_str_digit))
    return b_digits


# BEGIN TRAINING THE MODEL


NAME = "2-sum-{}-inputs-{}-largest-number-{}-neurons-{}-hidden-layers-{}-epochs-{}-samples-{}-batch-size-{}".format(
    LEN_NUMS, largest_bin_number, NEURONS, HIDDEN_LAYERS, EPOCHS, SAMPLES, BATCH_SIZE, int(time.time()))

FILENAME = "./model-development/{}.h5".format(NAME)

print()
print(NAME)
print()
# tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

model = Sequential()

model.add(Flatten(input_shape=[LEN_NUMS + 1, digits]))

for _ in range(HIDDEN_LAYERS):
    # Hidden layers
    model.add(Dense(NEURONS))
    model.add(Activation('relu'))

# Dropout layer
model.add(Dropout(0.2))

# Output Layer
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# callbacks=[tensorboard]
model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE,
          validation_split=0.1)

#Save model
model.save(FILENAME)
