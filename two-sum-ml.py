import math
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten

nums = [2, 7, 11, 15]
target = 9

def two_sum(nums, target):

    ### CREATE THE DATASET ###

    NEURONS = 256
    SAMPLES = 2000000
    BATCH_SIZE = 128
    HIDDEN_LAYERS = 3
    EPOCHS = 20

    SMALLEST_SAFE_NUMBER = 255  # 11111111
    LEN_NUMS = len(nums)

    largest_number = max(max(nums), target)
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
        print()
        print('creating dataset...')

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
            for q_nums in nums:
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

    ### TRAIN MODEL ###

    NAME = "2-sum-{}-inputs-{}-largest-number-{}-neurons-{}-hidden-layers-{}-epochs-{}-samples-{}-batch-size".format(
        LEN_NUMS, largest_bin_number, NEURONS, HIDDEN_LAYERS, EPOCHS, SAMPLES, BATCH_SIZE)

    # FILENAME = "{}.h5".format(NAME)
    # tensorboard = TensorBoard(log_dir='logs/{}'.format(NAME))

    print()
    print(NAME)
    print()

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
              validation_split=0.1, )

    # model.save(FILENAME)

    ### PREDICT ANSWER TO ORIGINAL PROBLEM ###

    def class_action(input_class):
        count_classes = 0
        iter = 0
        for j in range(LEN_NUMS - 1, 0, -1):
            if count_classes <= input_class and input_class < count_classes + j:
                diff = input_class - count_classes + 1
                return [iter, iter + diff]
            count_classes += j
            iter += 1

    class colors:
        ok = '\033[92m'
        fail = '\033[91m'
        close = '\033[0m'

    def make_prediction_non_train_data(target, nums):

        def output_one_answer():
            pred = model.predict(a)
            predicted_class = np.argmax(pred[0])
            print()
            print(pred) # The probabilities of all classes
            print('Predicted Class:', predicted_class) # The output class predicted
            ans_idx = class_action(predicted_class)
            pred_num1 = nums[ans_idx[0]]
            pred_num2 = nums[ans_idx[1]]
            print()

            english = "The NN believes {} + {} = {}".format(
                pred_num1, pred_num2, target)

            if pred_num1 + pred_num2 == target:
                print(english, colors.ok + '☑' + colors.close, end=' \n')
            else:
                print(english, colors.fail + '☒' + colors.close, end=' \n')
            return ans_idx

        def output_multiple_answers():
            pred = model.predict(a)
            answers = []
            for idx, class_conf in enumerate(pred[0]):
                # 0.01 is OK sometimes
                if class_conf > 0.01:
                    predicted_class = idx
                    print()
                    print(pred)
                    print('Predicted Class:', predicted_class)
                    ans_idx = class_action(predicted_class)
                    answers.append(ans_idx)
                    pred_num1 = nums[ans_idx[0]]
                    pred_num2 = nums[ans_idx[1]]
                    print()

                    english = "The NN believes {} + {} = {}".format(
                        pred_num1, pred_num2, target)

                    if pred_num1 + pred_num2 == target:
                        print(english, colors.ok + '☑' +
                              colors.close, end=' \n')
                    else:
                        print(english, colors.fail + '☒' +
                              colors.close, end=' \n')
            return answers

        nums_array = [bin_encode(target)]

        for i in range(LEN_NUMS):
            nums_array.append(bin_encode(nums[i]))

        a = np.array([nums_array])

        answer = output_one_answer()
        # If you want to get multiple answers, uncomment the below line
        # answer = output_multiple_answers()

        return answer

    return make_prediction_non_train_data(target, nums)


ans = two_sum(nums, target)
print(ans)
