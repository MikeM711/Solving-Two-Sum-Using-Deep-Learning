import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pickle
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

pickle_in = open("./model-development/X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("./model-development/y.pickle", "rb")
y = pickle.load(pickle_in)

MODEL_NAME = "2-sum-4-inputs-255-largest-number-256-neurons-3-hidden-layers-20-epochs-2000000-samples-128-batch-size-1587933661"
model = keras.models.load_model('./model-development/{}.h5'.format(MODEL_NAME))

SMALLEST_SAFE_NUMBER = 255  # 11111111
LEN_NUMS = len(tsq.nums)


def get_num_classes():
    num_classes = 0
    for i in range(LEN_NUMS - 1, 0, -1):
        num_classes += i
    return num_classes


# get the number of classes
num_classes = get_num_classes()

largest_number = max(max(tsq.nums), tsq.target)
largest_number = SMALLEST_SAFE_NUMBER if SMALLEST_SAFE_NUMBER > largest_number else largest_number + 1
digits = len("{0:b}".format(largest_number))


class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'


def bin_encode(x):
    bin_str = "{0:b}".format(x)
    if len(bin_str) != digits:
        num_zeroes = digits - len(bin_str)
        bin_str = '0'*num_zeroes + bin_str
    b_digits = []
    for b_str_digit in bin_str:
        b_digits.append(int(b_str_digit))
    return b_digits


def binary_arr_to_decimal(binary_arr):
    binary_num = ''.join(map(str, binary_arr))
    return int(binary_num, 2)


def class_action(input_class):
	count_classes = 0
	iter = 0
	for j in range(LEN_NUMS - 1, 0, -1):
		if count_classes <= input_class and input_class < count_classes + j:
			diff = input_class - count_classes + 1
			return [iter, iter + diff]
		count_classes += j
		iter += 1


def make_prediction_non_train_data(target, nums):
    nums_array = [bin_encode(target)]

    for i in range(LEN_NUMS):
        nums_array.append(bin_encode(nums[i]))

    a = np.array([nums_array])

    print()

    def output_one_answer():
        pred = model.predict(a)
        predicted_class = np.argmax(pred[0])
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
                    print(english, colors.ok + '☑' + colors.close, end=' \n')
                else:
                    print(english, colors.fail + '☒' + colors.close, end=' \n')
        return answers

    answer = output_one_answer()
    # If you want to get multiple answers, uncomment the below line
    # answer = output_multiple_answers()

    return answer


target = tsq.target if tsq.custom_target == None else tsq.custom_target
nums = tsq.nums if tsq.custom_nums == None else tsq.custom_nums

# Make a prediction based on inputs inside two_sum_question.py file
answer = make_prediction_non_train_data(target, nums)

print(answer)
