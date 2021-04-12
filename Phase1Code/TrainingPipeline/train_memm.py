from nltk.classify import MaxentClassifier
from nltk import pos_tag, word_tokenize
import pickle, re

dictionary = {}
test_sentences = []

def get_dict():
    with open("../DataFiles/dictionary.txt", "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line[:len(line)-1]
            items = line.split(",")
            dictionary[items[0]] = {"":items[1]}


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


def maxent_train(feature_list):
    labeled_features = []
    for (word, tag, shape, label, prev_word, prev_tag, prev_shape, previous_label) in feature_list:
        labeled_features.append((feature_template(
            word, tag, shape, prev_word, prev_tag, prev_shape, previous_label), label))
    f = open("../DataFiles/ner_labeler.pickle", "wb")
    maxent_classifier = MaxentClassifier.train(labeled_features, max_iter=40)
    pickle.dump(maxent_classifier, f)
    f.close()


def main():
    feature_list = []
    pattern = re.compile(r"\('(.*)', '(.*)', '(.*)', '(.*)', '(.*)', '(.*)', '(.*)', '(.*)'\)")
    with open("../DataFiles/memm_data.txt", "r") as f:
        lines = f.readlines()
        index = 1
        for line in lines:
            if index % 10000 == 0:
                print(index, "/", len(lines))
            index = index + 1
            line = line[:len(line) - 1]
            searched = pattern.search(line)
            feature_list.append(searched.groups())
    get_dict()
    maxent_train(feature_list) #trains the model on training data

main()