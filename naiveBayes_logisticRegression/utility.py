from re import sub
from os import scandir
from nltk.stem import SnowballStemmer

from Classifier import Classifier


def normalize_text(text):
    return sub("[^a-z]+", ' ', text.lower())


def read_data(directory_name):
    data = {}
    with scandir(directory_name) as data_directory:
        for class_entry in data_directory:
            if class_entry.is_dir():
                classification = class_entry.name
                if classification not in data:
                    data[classification] = []
                with scandir(class_entry.path) as class_directory:
                    for file_entry in class_directory:
                        if file_entry.is_file() and file_entry.name.endswith(".txt"):
                            with open(file_entry.path, 'r', encoding="utf8", errors="ignore") as file:
                                data[classification].append(normalize_text(file.read()))
    return data


def read_data_python35(directory_name):
    data = {}
    data_directory = scandir(directory_name)
    for class_entry in data_directory:
        if class_entry.is_dir():
            classification = class_entry.name
            if classification not in data:
                data[classification] = []
            class_directory = scandir(class_entry.path)
            for file_entry in class_directory:
                if file_entry.is_file() and file_entry.name.endswith(".txt"):
                    file = open(file_entry.path, 'r', encoding="utf8", errors="ignore")
                    data[classification].append(normalize_text(file.read()))
                    file.close()
    return data


def get_stop_words(stop_words_on):
    if stop_words_on:
        return [" ", "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
                "arent", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but",
                "by", "cant", "cannot", "could", "couldnt", "did", "didnt", "do", "does", "doesnt", "doing", "dont",
                "down", "during", "each", "few", "for", "from", "further", "had", "hadnt", "has", "hasnt", "have",
                "havent", "having", "he", "hed", "hell", "hes", "her", "here", "heres", "hers", "herself", "him",
                "himself", "his", "how", "hows", "i", "id", "ill", "im", "ive", "if", "in", "into", "is", "isnt",
                "it", "its", "its", "itself", "lets", "me", "more", "most", "mustnt", "my", "myself", "no", "nor",
                "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
                "over", "own", "same", "shant", "she", "shed", "shell", "shes", "should", "shouldnt", "so", "some",
                "such", "than", "that", "thats", "the", "their", "theirs", "them", "themselves", "then", "there",
                "theres", "these", "they", "theyd", "theyll", "theyre", "theyve", "this", "those", "through", "to",
                "too", "under", "until", "up", "very", "was", "wasnt", "we", "wed", "well", "were", "weve", "werent",
                "what", "whats", "when", "whens", "where", "wheres", "which", "while", "who", "whos", "whom", "why",
                "whys", "with", "wont", "would", "wouldnt", "you", "youd", "youll", "youre", "youve", "your", "yours",
                "yourself", "yourselves"]
    else:
        return [" "]


def data_to_classifiers(data, filter_stop_words):
    classifiers = []
    stop_words = get_stop_words(filter_stop_words)
    stemmer = SnowballStemmer("english")
    for classification, text_files in data.items():
        for text in text_files:
            classifier_new = Classifier(classification)
            classifier_new.count_words(stop_words, stemmer, text)
            classifiers.append(classifier_new)
    return classifiers
