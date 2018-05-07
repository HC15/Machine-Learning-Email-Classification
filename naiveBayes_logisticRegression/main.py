from sys import argv

from utility import read_data
import naive_bayes
import logistic_regression


def main():
    training = read_data(argv[1])
    test = read_data(argv[2])

    classifiers_unfiltered = naive_bayes.train(training, False)
    accuracy_unfiltered_nb = naive_bayes.test(test, False, classifiers_unfiltered)
    print("Naive Bayes is", "{0:.6f}".format(accuracy_unfiltered_nb), "accurate with stop words unfiltered")

    classifiers_filtered = naive_bayes.train(training, True)
    accuracy_filtered_nb = naive_bayes.test(test, True, classifiers_filtered)
    print("Naive Bayes is", "{0:.6f}".format(accuracy_filtered_nb), "accurate with stop words filtered")

    for i in range(3, len(argv)):
        print()
        lambda_constant = float(argv[i])

        weights_unfiltered = logistic_regression.train(training, False, 25, lambda_constant)
        accuracy_unfiltered_lr = logistic_regression.test(test, False, weights_unfiltered)
        print("Logistic Regression is", "{0:.6f}".format(accuracy_unfiltered_lr),
              "accurate with stop words unfiltered and lambda constant equal to", lambda_constant)

        weights_filtered = logistic_regression.train(training, True, 25, lambda_constant)
        accuracy_filtered_lr = logistic_regression.test(test, True, weights_filtered)
        print("Logistic Regression is", "{0:.6f}".format(accuracy_filtered_lr),
              "accurate with stop words filtered and lambda constant equal to", lambda_constant)


if __name__ == "__main__":
    main()
