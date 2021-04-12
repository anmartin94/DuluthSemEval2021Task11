from nltk.classify import MaxentClassifier
from nltk import pos_tag, word_tokenize
import pickle, re

dictionary = {}

def get_dict():
    with open("../DataFiles/dictionary.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line[:len(line)-1]
            items = line.split(",")
            dictionary[items[0]] = {"":items[1]}


def get_word_shape(word):
    shape = ""
    for c in word:
        if c.isalpha():
            if c.isupper():
                if shape == "":
                    shape = "X"
                    continue
                if shape[:len(shape)] != "X":
                    shape = shape + "X"
            elif c.islower():
                if shape == "":
                    shape = "x"
                    continue
                if shape[len(shape)-1:len(shape)] != "x":
                    shape = shape + "x"
        elif c.isnumeric():
            if shape == "":
                shape = "d"
                continue
            if shape[len(shape)-1:len(shape)] != "d":
                shape = shape + "d"
        else:
            shape = shape + c
    return shape


def feature_template(word, tag, shape, prev_word, prev_tag, prev_shape, prev_label):
    features = {}
    features['identity'] = word
    features['pos'] = tag
    features['shape'] = shape
    features['prev_identity'] = prev_word
    features['prev_pos'] = prev_tag
    features['prev_shape'] = prev_shape
    features['previous_label'] = prev_label
    return features


def initialize_lists(wordlist):
    viterbi = []
    for i in range(len(wordlist) + 2):
        sublist = []
        for j in range(len(wordlist) + 2):
            sublist.append(0)
        viterbi.append(sublist)
    backpointer = []
    for i in range(len(wordlist) + 2):
        sublist = []
        for j in range(len(wordlist) + 2):
            sublist.append("")
        backpointer.append(sublist)
    return viterbi, backpointer

#Performs the Viterbi algorithm to optimize the sequence of tags.
#Follows the algorithm outlined on page 153 of Jurafsky and Manning's
#Speech and Natural Language Processing.
def memm_viterbi(maxent_classifier, test_sentences):
    labels = ["B", "I", "O"]
    wordlist = []
    for sentence in test_sentences:
        words = pos_tag(word_tokenize(sentence))
        for word in words:
            if not word[0].isalnum():
                #words.remove(word)
                continue
            wordlist.append(word)
    if len(wordlist) == 0:
        return [], []
    first_word = wordlist[0][0]
    first_pos_tag = wordlist[0][1]
    first_shape = get_word_shape(wordlist[0][0])
    viterbi, backpointer = initialize_lists(wordlist)
    #For the first word in the sentence
    for l in range(len(labels)):
        probability = maxent_classifier.prob_classify(feature_template(
            first_word, first_pos_tag, first_shape, "", "", "", ""))
        posterior = float(probability.prob(labels[l]))
        viterbi[l][1] = posterior
        backpointer[l][1] = 0
        prev_word = first_word
        prev_tag = first_pos_tag
        prev_shape = first_shape
    #For all other words
    for w in range(1, len(wordlist)):
        for l in range(len(labels)):
            word = wordlist[w][0]
            tag = wordlist[w][1]
            probability = maxent_classifier.prob_classify(feature_template(
                word, tag, get_word_shape(word), prev_word, prev_tag, prev_shape, labels[0]))
            prev_word = word
            prev_tag = tag
            prev_shape = get_word_shape(word)
            posterior = float(probability.prob(labels[l]))
            max_viterbi = float(viterbi[0][w]) * posterior
            max_previous_state = 0
            for i in range(1, len(labels)):
                word = wordlist[w][0]
                tag = wordlist[w][1]
                probability = maxent_classifier.prob_classify(feature_template(
                    word, tag, get_word_shape(word), prev_word, prev_tag, prev_shape, labels[i]))
                posterior = float(probability.prob(labels[l]))
                if float(viterbi[i][w]) * posterior > max_viterbi:
                    max_viterbi = float(viterbi[i][w]) * posterior
                    max_previous_state = i
            viterbi[l][w + 1] = max_viterbi
            backpointer[l][w + 1] = labels[max_previous_state]
    max_probability = float(viterbi[0][len(wordlist)]) * float(dictionary[labels[0]][""])
    max_previous_state = 0
    for i in range(1, len(labels)):
        if float(viterbi[i][len(wordlist)]) * float(dictionary[labels[i]][""]) > max_probability:
            max_probability = float(viterbi[i][len(wordlist)]) * float(dictionary[labels[i]][""])

            max_previous_state = i
    viterbi[len(labels)-1][len(wordlist) + 1] = max_probability
    backpointer[len(labels)-1][len(wordlist) + 1] = labels[max_previous_state]
    path_reverse = [labels[max_previous_state]]
    max_previous_tag = labels[max_previous_state]

    i = 0
    while i < (len(wordlist) - 1):
        path_reverse.append(backpointer[labels.index(max_previous_tag)][len(wordlist) - i])
        max_previous_tag = backpointer[labels.index(max_previous_tag)][len(wordlist) - i]
        i = i + 1

    index = len(path_reverse)
    path = []
    while index >= 1:
        path.append(path_reverse[index - 1])
        index = index - 1
    return wordlist, path



def main(data):
    get_dict()
    model = open("../DataFiles/ner_labeler.pickle", "rb")
    maxent_classifier = pickle.load(model)
    index = 1
    for path in data:
        print(index, "/", len(data))
        index = index + 1
        for sentence_index in data[path]:
            wordlist, sentence_path = memm_viterbi(maxent_classifier, [data[path][sentence_index][1]])
            labeled_sentence = []
            for i in range(len(wordlist)):
                labeled_sentence.append([wordlist[i][0], sentence_path[i]])
            data[path][sentence_index].append(labeled_sentence)
    return data

