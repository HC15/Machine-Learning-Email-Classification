class Classifier:

    def __init__(self, classification):
        self.classification = classification
        self.file_count = 0
        self.word_counter = {}
        self.word_total = 0

    def count_words(self, stop_words, stemmer, text):
        self.file_count += 1
        text_split = text.split(' ')
        for word in text_split:
            if len(word) > 0 and word not in stop_words:
                word_stem = stemmer.stem(word)
                if word_stem not in self.word_counter:
                    self.word_counter[word_stem] = 1
                else:
                    self.word_counter[word_stem] += 1
                self.word_total += 1

    def laplace_smoothing(self, words, strength, classes):
        for word in words:
            if word not in self.word_counter:
                self.word_counter[word] = strength
            else:
                self.word_counter[word] += strength
            self.word_total += (strength * classes)
