from math import exp

from utility import data_to_classifiers


def conditional_probability(weights, word_counter, classification):
    exponent = weights["Weight 0"]
    for word, count in word_counter.items():
        if word in weights:
            exponent += (weights[word] * count)

    numerator = 1.0
    if classification == "spam":
        numerator = exp(exponent)
    return numerator / (1.0 + exp(exponent))


def train(data, filter_stop_words, iteration_limit, lambda_constant):
    classifiers = data_to_classifiers(data, filter_stop_words)

    weights = {"Weight 0": 0.0}
    for classifier in classifiers:
        for word in classifier.word_counter.keys():
            if word not in weights:
                weights[word] = 0.0

    for iteration in range(iteration_limit):
        dw = dict.fromkeys(list(weights.keys()), 0.0)
        for word, weight in dw.items():
            for classifier in classifiers:
                class_value = 0.0
                if classifier.classification == "spam":
                    class_value = 1.0
                if word in classifier.word_counter:
                    dw[word] += classifier.word_counter[word] * \
                                (class_value - conditional_probability(weights, classifier.word_counter, "spam"))
        for word, weight in weights.items():
            weights[word] += (0.008 * (dw[word] - (lambda_constant * weight)))
    return weights


def test(data, filter_stop_words, weights):
    classifiers = data_to_classifiers(data, filter_stop_words)

    correct = 0
    for classifier in classifiers:
        classification_rule = weights["Weight 0"]
        for word, count in classifier.word_counter.items():
            if word in weights:
                classification_rule += (weights[word] * count)

        if classification_rule > 0.0:
            if classifier.classification == "spam":
                correct += 1
        else:
            if classifier.classification == "ham":
                correct += 1
    return correct / len(classifiers)
