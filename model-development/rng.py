import random
import math

# Create a randomly generated 2 sum question

inputs = 5
MAX_NUM = 255
upper_bound = MAX_NUM/2

def get_num_classes():
    num_classes = 0
    for i in range(inputs - 1, 0, -1):
        num_classes += i
    return num_classes

num_classes = get_num_classes()

two_sum_nums_array_sample = []

for idx in range(inputs):
    two_sum_nums_array_sample.append(
        math.floor(random.random() * upper_bound))

# Get a random class (indexed at 0)
# Example: class of 0 corresponds to answer [0, 1]
random_class = math.floor(random.random() * num_classes)

# Get a target using the associated class

count_classes = 0
iter = 0
for j in range(inputs - 1, 0, -1):
    if count_classes <= random_class and random_class < count_classes + j:
        diff = random_class - count_classes + 1
        sample_target = two_sum_nums_array_sample[iter] + \
            two_sum_nums_array_sample[iter + diff]
        break
    count_classes += j
    iter += 1

print()
print('nums =',two_sum_nums_array_sample)
print('target =', sample_target)
