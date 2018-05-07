from sys import maxsize
from math import log10
from nltk.stem import SnowballStemmer

from Classifier import Classifier
from utility import get_stop_words


def train(data, filter_stop_words):
    classifiers = []
    stop_words = get_stop_words(filter_stop_words)
    stemmer = SnowballStemmer("english")
    for classification, text_files in data.items():
        classifier = Classifier(classification)
        for text in text_files:
            classifier.count_words(stop_words, stemmer, text)
        classifiers.append(classifier)
    return classifiers


def test(data, filter_stop_words, classifiers):
    file_total = 0
    for classifier in classifiers:
        file_total += classifier.file_count

    correct = 0
    stop_words = get_stop_words(filter_stop_words)
    stemmer = SnowballStemmer("english")
    for classification, text_files in data.items():
        for text in text_files:
            text_split = text.split(' ')
            words = set()
            for word in text_split:
                if len(word) > 0 and word not in stop_words:
                    word_stem = stemmer.stem(word)
                    words.add(word_stem)

            predicted_class = ""
            highest_probability = -maxsize - 1
            for classifier in classifiers:
                probability = log10(classifier.file_count / file_total)
                classifier_smooth = classifier
                classifier_smooth.laplace_smoothing(words, 1, len(classifiers))
                for word in text_split:
                    if len(word) > 0 and word not in stop_words:
                        word_stem = stemmer.stem(word)
                        probability += log10(classifier_smooth.word_counter[word_stem] / classifier_smooth.word_total)

                if probability > highest_probability:
                    highest_probability = probability
                    predicted_class = classifier_smooth.classification

            if predicted_class == classification:
                correct += 1
    return correct / file_total
