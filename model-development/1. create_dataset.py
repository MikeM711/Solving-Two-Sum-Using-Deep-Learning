import random
import math
import numpy as np
import pickle
from tensorflow.keras.utils import to_categorical
import two_sum_question as tsq

'''

Example of a Two-Sum question
input = [1, 2, 3, 10]
target = 5
Answer: 2 + 3 = 5; therefore output = [1, 2] (class 3 below)

Answer key:

Class | Answers  |       Output
0 | [1, 1, 0, 0] | [1, 0, 0, 0, 0, 0]
1 | [1, 0, 1, 0] | [0, 1, 0, 0, 0, 0]
2 | [1, 0, 0, 1] | [0, 0, 1, 0, 0, 0]
3 | [0, 1, 1, 0] | [0, 0, 0, 1, 0, 0]
4 | [0, 1, 0, 1] | [0, 0, 0, 0, 1, 0]
5 | [0, 0, 1, 1] | [0, 0, 0, 0, 0, 1]

'''


SMALLEST_SAFE_NUMBER = 255  # 11111111
SAMPLES = 2000000
LEN_NUMS = len(tsq.nums)

largest_number = max(max(tsq.nums), tsq.target)
largest_safe_number = SMALLEST_SAFE_NUMBER if SMALLEST_SAFE_NUMBER > largest_number else largest_number + 1
digits = len("{0:b}".format(largest_safe_number))
largest_bin_number = int('1'*digits, 2)

upper_bound = math.floor(largest_bin_number/2)


def get_num_classes():
    num_classes = 0
    for i in range(LEN_NUMS - 1, 0, -1):
        num_classes += i
    return num_classes


# get the number of classes aka possible "answers"
num_classes = get_num_classes()


def bin_encode(x):
    bin_str = "{0:b}".format(x)
    if len(bin_str) != digits:
        num_zeroes = digits - len(bin_str)
        bin_str = '0'*num_zeroes + bin_str
    b_digits = []
    for b_str_digit in bin_str:
        b_digits.append(int(b_str_digit))
    return b_digits


def binary_arr_to_decimal(binary):
    return int(binary, 2)


def create_dataset():
    print('creating dataset...')
    # Dataset
    X = []
    y = []

    i = 0

    while i < SAMPLES:
        if i % math.floor(SAMPLES/10) == 0:
            print('Created {} out of {} Samples'.format(i, SAMPLES))

        # Create a "nums" array sample

        # sample array will have the same number of items as question
        # sample array will have a target and nums where the binary equivalent
        # does NOT have more digits than the MAX binary digits as the question

        # (I have set the smallest MAX digit to be 255 - binary: 11111111)

        two_sum_nums_array_sample = []
        for idx in range(LEN_NUMS):
            two_sum_nums_array_sample.append(
                math.floor(random.random() * upper_bound))

        # Get a random class (indexed at 0)
        # Example: class of 0 corresponds to answer [0, 1]
        random_class = math.floor(random.random() * num_classes)

        # Get a target using the associated class

        count_classes = 0
        iter = 0
        for j in range(LEN_NUMS - 1, 0, -1):
            if count_classes <= random_class and random_class < count_classes + j:
                diff = random_class - count_classes + 1
                sample_target = two_sum_nums_array_sample[iter] + \
                    two_sum_nums_array_sample[iter + diff]
                break
            count_classes += j
            iter += 1

        # Two sum can ONLY HAPPEN ONCE - per leetcode

        num_two_sum_answers = 0
        two_sum_hash = {}

        for j in range(len(two_sum_nums_array_sample)):
            num = two_sum_nums_array_sample[j]
            if sample_target - num in two_sum_hash:
                num_two_sum_answers += 1
                two_sum_hash[num] = j
            else:
                two_sum_hash[num] = j

        # For the very slim chance we have randomly generated a two_sum_nums_array_sample
        # that is exactly like nums in the question - that's like ... mildly cheating

        identical_to_nums = True
        for q_nums in tsq.nums:
            if q_nums in two_sum_hash:
                del two_sum_hash[q_nums]
            else:
                identical_to_nums = False
                break

        if num_two_sum_answers == 1 and identical_to_nums == False:

            # The sample array generated is a good sample for the network to learn from

            bin_target = bin_encode(sample_target)
            X_bin = [bin_target]

            for sample_num in two_sum_nums_array_sample:
                X_bin.append(bin_encode(sample_num))

            X.append(X_bin)
            y.append([random_class])
            i += 1

    print('Finished: Created {} out of {} Samples'.format(SAMPLES, SAMPLES))

    # one hot encode y
    return np.array(X), to_categorical(np.array(y), num_classes)


X, y = create_dataset()

# put dataset into pickle to feed into model
print('done')

pickle_out = open("./model-development/X.pickle", "wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("./model-development/y.pickle", "wb")
pickle.dump(y, pickle_out)
pickle_out.close()
