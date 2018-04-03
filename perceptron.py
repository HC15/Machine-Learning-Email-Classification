def output(text_file, weights):
    activation = weights[" BIAS "]
    for word, count in text_file.word_counter.items():
        if word in weights:
            activation += weights[word] * count
    if activation > 0:
        return 1
    else:
        return -1


def train(data, iteration_limit, learning_rate):
    weights = {" BIAS ": 1.0}
    for iteration in range(iteration_limit):
        for text_file in data:
            if text_file.spam:
                target_value = 1
            else:
                target_value = -1
            perceptron_output = output(text_file, weights)
            for word, count in text_file.word_counter.items():
                if word not in weights:
                    weights[word] = 0.0
                weights[word] += learning_rate * (target_value - perceptron_output) * count
    return weights


def test(data, weights):
    correct = 0
    for text_file in data:
        perceptron_output = output(text_file, weights)
        if perceptron_output == 1 and text_file.spam:
            correct += 1
        elif perceptron_output == -1 and not text_file.spam:
            correct += 1
    return correct / len(data)
