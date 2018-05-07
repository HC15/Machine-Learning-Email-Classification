from re import sub
from os import scandir

from nltk.stem import SnowballStemmer

from TextFile import TextFile


def get_stop_words():
    return ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "arent",
            "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "cant",
            "cannot", "could", "couldnt", "did", "didnt", "do", "does", "doesnt", "doing", "dont", "down", "during",
            "each", "few", "for", "from", "further", "had", "hadnt", "has", "hasnt", "have", "havent", "having", "he",
            "hed", "hell", "hes", "her", "here", "heres", "hers", "herself", "him", "himself", "his", "how", "hows",
            "i", "id", "ill", "im", "ive", "if", "in", "into", "is", "isnt", "it", "its", "its", "itself", "lets",
            "me", "more", "most", "mustnt", "my", "myself", "no", "nor", "not", "of", "off", "on", "once", "only",
            "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shant", "she", "shed",
            "shell", "shes", "should", "shouldnt", "so", "some", "such", "than", "that", "thats", "the", "their",
            "theirs", "them", "themselves", "then", "there", "theres", "these", "they", "theyd", "theyll", "theyre",
            "theyve", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasnt", "we",
            "wed", "well", "were", "weve", "werent", "what", "whats", "when", "whens", "where", "wheres", "which",
            "while", "who", "whos", "whom", "why", "whys", "with", "wont", "would", "wouldnt", "you", "youd", "youll",
            "youre", "youve", "your", "yours", "yourself", "yourselves"]


def normalize_text(text):
    return sub("[^a-z]+", ' ', text.lower())


def read_data(directory_name):
    data = {"unfiltered": [], "filtered": []}
    stemmer = SnowballStemmer("english")
    stop_words = get_stop_words()
    with scandir(directory_name) as data_directory:
        for class_entry in data_directory:
            if class_entry.is_dir():
                spam = class_entry.name == "spam"
                with scandir(class_entry.path) as class_directory:
                    for file_entry in class_directory:
                        if file_entry.is_file() and file_entry.name.endswith(".txt"):
                            with open(file_entry.path, 'r', encoding="utf8", errors="ignore") as file:
                                text_file_unfiltered = TextFile(spam)
                                text_file_filtered = TextFile(spam)
                                text_split = normalize_text(file.read()).split(' ')
                                for word in text_split:
                                    if len(word) > 0 and word != " ":
                                        word_stem = stemmer.stem(word)

                                        if word_stem not in text_file_unfiltered.word_counter:
                                            text_file_unfiltered.word_counter[word_stem] = 1
                                        else:
                                            text_file_unfiltered.word_counter[word_stem] += 1

                                        if word_stem not in stop_words:
                                            if word_stem not in text_file_filtered.word_counter:
                                                text_file_filtered.word_counter[word_stem] = 1
                                            else:
                                                text_file_filtered.word_counter[word_stem] += 1
                                data["unfiltered"].append(text_file_unfiltered)
                                data["filtered"].append(text_file_filtered)
    return data
