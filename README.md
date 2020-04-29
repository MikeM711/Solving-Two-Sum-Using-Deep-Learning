# Solving-Two-Sum-Using-Deep-Learning

Amused by [Joel Grus - Fizz Buss in Tensorflow](https://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/), I have set out to create my own neural network for solving a traditionally simple problem that is also uncharted in the machine learning world. 

The chosen question: [Two Sum](https://leetcode.com/problems/two-sum/) (spoiler: just use a hashtable).

A video of the [Neural Network in action](https://www.youtube.com/watch?v=YYgJHANyeUM)

## So... What's the time complexity?

<img width=300px src="https://raw.githubusercontent.com/MikeM711/Solving-Two-Sum-Using-Deep-Learning/master/repo-img/the-office.png"/>

## Repo Files

* `two-sum-ml.py`: The answer to your interviewer's question. Enter your nums and target, and run (both the program, and away from the angry interviewer)
* `model-development` files: If you wish to poke around, or help make the network more robust, inside this folder are 3 numbered files - a file for collecting data, training the model and testing the model. These three files have parameters that are dictated by the `two_sum_question.py` file (similar to how the question dictates `two-sum-ml.py` parameters)
* `generated-models` files: 8 models that I have created using the above 3 files.
* `generated-model-logs` files: Tensorboard logs for the above 8 files.
* `jupyter-notebook` file: A faster way to test models.

## About The Two Sum Network

When it comes to the Two Sum problem, as more num items are added to the array, there are exponentially larger possible answers - this can make learning very tricky for questions involving a large number of inputs. For the current neural network, it is very good at solving problems involving 4 items (which was my original goal), and decent with 6 items. As more items are added, the network has a trickier time. Upping the "max possible number" has shown to exacerbate the network's issues with how it can predict answers.

### The "Good" models

For questions that contain: 

* 4 array inputs and include numbers less than 255, the validation accuracy is `98.34%`.
* 4 array inputs and include numbers less than 16,383, the validation accuracy is `94.92%`.
* 4 array inputs and include numbers less than 1,048,575, the validation accuracy is `95.28`.

### The "OK" models

For questions that contain:

* 6 array inputs and include numbers less than 255, the validation accuracy is `88.28%`.
* 6 array inputs and include numbers less than 16,383, the validation accuracy is `83.04%`.
* 8 array inputs and include numbers less than 255, the validation accuracy is `73.04%`.
* 8 array inputs and include numbers less than 16,383, the validation accuracy is `54.05%`.

### The "Bad" models

For questions that contain:

* 26 array inputs and include numbers less than 255, the validation accuracy is ` 0.03%` (most likely due to the fact that there are 325 possible answers to learn from at this point)


