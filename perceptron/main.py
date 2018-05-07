from sys import argv

from utility import read_data
import perceptron


def main():
    training = read_data(argv[1])
    test = read_data(argv[2])

    filter_modes = ["unfiltered", "filtered"]
    iteration_limits = [10, 25, 50, 100]
    training_rates = [0.01, 0.05, 0.1, 0.5, 0.8]
    for mode in filter_modes:
        for limit in iteration_limits:
            for rate in training_rates:
                weights = perceptron.train(training[mode], limit, rate)
                accuracy = perceptron.test(test[mode], weights) * 100
                print("{0:.6f}".format(accuracy) + "% accurate:", mode, "stop words,",
                      limit, "iterations,", rate, "training rate")


if __name__ == "__main__":
    main()
