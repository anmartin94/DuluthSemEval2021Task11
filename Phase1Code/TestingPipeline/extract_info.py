import os
import re
import shutil

from nltk import word_tokenize

input_data = {'evaluation-phase1-master/data-to-text_generation/4/1809.00582v2-Stanza-out.txt': {'1': ['0', 'title', [['title', 'O']], [{'dep': 'ROOT', 'governor': 0, 'governorGloss': 'ROOT', 'dependent': 1, 'dependentGloss': 'title'}]], '2': ['11', 'Data - to - Text Generation with Content Selection and Planning', [['Data', 'B'], ['to', 'B'], ['Text', 'B'], ['Generation', 'B'], ['with', 'B'], ['Content', 'B'], ['Selection', 'B'], ['and', 'B'], ['Planning', 'B']], [{'dep': 'ROOT', 'governor': 0, 'governorGloss': 'ROOT', 'dependent': 3, 'dependentGloss': 'Text'}, {'dep': 'nsubj', 'governor': 3, 'governorGloss': 'Text', 'dependent': 1, 'dependentGloss': 'Data'}, {'dep': 'mark', 'governor': 3, 'governorGloss': 'Text', 'dependent': 2, 'dependentGloss': 'to'}, {'dep': 'obj', 'governor': 3, 'governorGloss': 'Text', 'dependent': 4, 'dependentGloss': 'Generation'}, {'dep': 'case', 'governor': 7, 'governorGloss': 'Selection', 'dependent': 5, 'dependentGloss': 'with'}, {'dep': 'compound', 'governor': 7, 'governorGloss': 'Selection', 'dependent': 6, 'dependentGloss': 'Content'}, {'dep': 'obl', 'governor': 3, 'governorGloss': 'Text', 'dependent': 7, 'dependentGloss': 'Selection'}, {'dep': 'cc', 'governor': 9, 'governorGloss': 'Planning', 'dependent': 8, 'dependentGloss': 'and'}, {'dep': 'conj', 'governor': 7, 'governorGloss': 'Selection', 'dependent': 9, 'dependentGloss': 'Planning'}]], '3': ['0', 'abstract', [['abstract', 'O']], [{'dep': 'ROOT', 'governor': 0, 'governorGloss': 'ROOT', 'dependent': 1, 'dependentGloss': 'abstract'}]], '4': ['0', 'Recent advances in data - to - text generation have led to the use of large - scale datasets and neural network models which are trained end - to - end , without explicitly modeling what to say and in what order .', [['Recent', 'O'], ['advances', 'O'], ['in', 'O'], ['data', 'O'], ['to', 'O'], ['text', 'O'], ['generation', 'O'], ['have', 'O'], ['led', 'O'], ['to', 'O'], ['the', 'O'], ['use', 'O'], ['of', 'O'], ['large', 'B'], ['scale', 'O'], ['datasets', 'O'], ['and', 'O'], ['neural', 'B'], ['network', 'B'], ['models', 'O'], ['which', 'O'], ['are', 'O'], ['trained', 'O'], ['end', 'O'], ['to', 'O'], ['end', 'O'], ['without', 'O'], ['explicitly', 'O'], ['modeling', 'O'], ['what', 'O'], ['to', 'O'], ['say', 'O'], ['and', 'O'], ['in', 'O'], ['what', 'O'], ['order', 'O']], [{'dep': 'ROOT', 'governor': 0, 'governorGloss': 'ROOT', 'dependent': 9, 'dependentGloss': 'led'}, {'dep': 'amod', 'governor': 2, 'governorGloss': 'advances', 'dependent': 1, 'dependentGloss': 'Recent'}, {'dep': 'nsubj', 'governor': 9, 'governorGloss': 'led', 'dependent': 2, 'dependentGloss': 'advances'}, {'dep': 'case', 'governor': 4, 'governorGloss': 'data', 'dependent': 3, 'dependentGloss': 'in'}, {'dep': 'nmod', 'governor': 2, 'governorGloss': 'advances', 'dependent': 4, 'dependentGloss': 'data'}, {'dep': 'case', 'governor': 7, 'governorGloss': 'generation', 'dependent': 5, 'dependentGloss': 'to'}, {'dep': 'compound', 'governor': 7, 'governorGloss': 'generation', 'dependent': 6, 'dependentGloss': 'text'}, {'dep': 'nmod', 'governor': 2, 'governorGloss': 'advances', 'dependent': 7, 'dependentGloss': 'generation'}, {'dep': 'aux', 'governor': 9, 'governorGloss': 'led', 'dependent': 8, 'dependentGloss': 'have'}, {'dep': 'case', 'governor': 12, 'governorGloss': 'use', 'dependent': 10, 'dependentGloss': 'to'}, {'dep': 'det', 'governor': 12, 'governorGloss': 'use', 'dependent': 11, 'dependentGloss': 'the'}, {'dep': 'obl', 'governor': 9, 'governorGloss': 'led', 'dependent': 12, 'dependentGloss': 'use'}, {'dep': 'case', 'governor': 16, 'governorGloss': 'datasets', 'dependent': 13, 'dependentGloss': 'of'}, {'dep': 'amod', 'governor': 15, 'governorGloss': 'scale', 'dependent': 14, 'dependentGloss': 'large'}, {'dep': 'compound', 'governor': 16, 'governorGloss': 'datasets', 'dependent': 15, 'dependentGloss': 'scale'}, {'dep': 'nmod', 'governor': 12, 'governorGloss': 'use', 'dependent': 16, 'dependentGloss': 'datasets'}, {'dep': 'cc', 'governor': 20, 'governorGloss': 'models', 'dependent': 17, 'dependentGloss': 'and'}, {'dep': 'amod', 'governor': 19, 'governorGloss': 'network', 'dependent': 18, 'dependentGloss': 'neural'}, {'dep': 'compound', 'governor': 20, 'governorGloss': 'models', 'dependent': 19, 'dependentGloss': 'network'}, {'dep': 'conj', 'governor': 16, 'governorGloss': 'datasets', 'dependent': 20, 'dependentGloss': 'models'}, {'dep': 'nsubj:pass', 'governor': 23, 'governorGloss': 'trained', 'dependent': 21, 'dependentGloss': 'which'}, {'dep': 'aux:pass', 'governor': 23, 'governorGloss': 'trained', 'dependent': 22, 'dependentGloss': 'are'}, {'dep': 'acl:relcl', 'governor': 16, 'governorGloss': 'datasets', 'dependent': 23, 'dependentGloss': 'trained'}, {'dep': 'obj', 'governor': 23, 'governorGloss': 'trained', 'dependent': 24, 'dependentGloss': 'end'}, {'dep': 'case', 'governor': 26, 'governorGloss': 'end', 'dependent': 25, 'dependentGloss': 'to'}, {'dep': 'obl', 'governor': 23, 'governorGloss': 'trained', 'dependent': 26, 'dependentGloss': 'end'}, {'dep': 'punct', 'governor': 23, 'governorGloss': 'trained', 'dependent': 27, 'dependentGloss': ','}, {'dep': 'case', 'governor': 30, 'governorGloss': 'modeling', 'dependent': 28, 'dependentGloss': 'without'}, {'dep': 'advmod', 'governor': 30, 'governorGloss': 'modeling', 'dependent': 29, 'dependentGloss': 'explicitly'}, {'dep': 'advcl', 'governor': 23, 'governorGloss': 'trained', 'dependent': 30, 'dependentGloss': 'modeling'}, {'dep': 'obj', 'governor': 30, 'governorGloss': 'modeling', 'dependent': 31, 'dependentGloss': 'what'}, {'dep': 'mark', 'governor': 33, 'governorGloss': 'say', 'dependent': 32, 'dependentGloss': 'to'}, {'dep': 'xcomp', 'governor': 30, 'governorGloss': 'modeling', 'dependent': 33, 'dependentGloss': 'say'}, {'dep': 'cc', 'governor': 37, 'governorGloss': 'order', 'dependent': 34, 'dependentGloss': 'and'}, {'dep': 'case', 'governor': 37, 'governorGloss': 'order', 'dependent': 35, 'dependentGloss': 'in'}, {'dep': 'det', 'governor': 37, 'governorGloss': 'order', 'dependent': 36, 'dependentGloss': 'what'}, {'dep': 'conj', 'governor': 33, 'governorGloss': 'say', 'dependent': 37, 'dependentGloss': 'order'}, {'dep': 'punct', 'governor': 9, 'governorGloss': 'led', 'dependent': 38, 'dependentGloss': '.'}]]}}

iu_dict = {"1": "ablation-analysis",
           "2": "approach",
           "3": "baselines",
           "4": "code",
           "5": "dataset",
           "6": "experimental-setup",
           "7": "experiments",
           "8": "hyperparameters",
           "9": "model",
           "11": "research-problem",
           "12": "results",
           "13": "tasks"
           }


def phrase_in_entities(phrase, labeled_words):
    '''words = word_tokenize(phrase)
    for word in words:
        if word in labeled_words.keys():
            if labeled_words[word] != "O":
                return True
    return False'''
    phrase = word_tokenize(phrase)
    for pair in labeled_words:
        if pair[1] != "O" and pair[0] in phrase:
            return True
    return False


def get_dep_dependencies(dependencies, phrase, core):
    for word in phrase:
        if word == core:
            continue
        for dep in dependencies:
            if dep['governor'] == word:
                phrase.append(dep['dependent'])
    phrase.sort()
    return phrase


def get_root_subject(dependencies, root_index):
    subjects = ['nsubj', 'nsubj:pass', 'compound']
    for dep in dependencies:
        if dep['dep'] in subjects:
            if dep['governor'] >= root_index:
                return dep['dependent']


def get_secondary_verb(dependencies, subj_index):
    verbs = ['acl:relcl']
    for dep in dependencies:
        if dep['dep'] in verbs:
            if dep['governor'] == subj_index:
                return dep['dependent']
    return None


def get_root_object(dependencies, root_index):
    #objects = ['obj', 'iobj', 'obl', 'dep', 'xcomp']
    for dep in dependencies:
        if dep['dependent'] < root_index:
            continue
        if dep['governor'] == root_index:
            #if dep['dep'] in objects:
            return dep['dependent']


def get_subject_phrase(dependencies, noun_index):
    #np_deps = ['advmod', 'amod', 'compound', 'nmod', 'nummod', 'nummod:gov', 'mark']
    phrase = [noun_index]
    for dep in dependencies:
        if dep['dependent'] == noun_index:
            return get_dep_dependencies(dependencies, phrase, noun_index)
        if dep['governor'] == noun_index:
            #if dep['dep'] in np_deps:
            phrase.append(dep['dependent'])
    return get_dep_dependencies(dependencies, phrase, noun_index)


def get_verb_phrase(dependencies, verb_index):
    phrase = [verb_index]
    for dep in dependencies:
        if dep['dependent'] == verb_index:
            return get_dep_dependencies(dependencies, phrase, verb_index)
        if dep['governor'] == verb_index:
            phrase.append(dep['dependent'])
    return get_dep_dependencies(dependencies, phrase, verb_index)


def get_object_phrase(dependencies, noun_index, verb_phrase):
    #np_deps = ['amod', 'compound', 'nmod', 'nummod', 'nummod:gov']
    preposition = ['case']
    noun_phrase = [noun_index]
    for dep in dependencies:
        if dep['dependent'] == noun_index:
            noun_phrase = get_dep_dependencies(dependencies, noun_phrase, noun_index)
            verb_phrase.sort()
            return noun_phrase, verb_phrase
        if dep['governor'] == noun_index:
            #if dep['dep'] in np_deps:
            noun_phrase.append(dep['dependent'])
            if dep['dep'] in preposition:
                verb_phrase.append(dep['dependent'])
    noun_phrase = get_dep_dependencies(dependencies, noun_phrase, noun_index)
    verb_phrase.sort()
    return noun_phrase, verb_phrase


def get_string_from_phrase(dependencies, phrase, sentence, sentence_index):
    s = str(sentence_index)
    last_word = phrase[len(phrase)-1]
    print(type(get_start_end_indices(sentence, phrase[0], last_word, get_word_from_index(dependencies, last_word))))
    start, end = get_start_end_indices(sentence, phrase[0], last_word, get_word_from_index(dependencies, last_word))
    s = s + "\t" + str(start) + "\t" + str(end) + "\t"
    for word_index in phrase:
        s = s + get_word_from_index(dependencies, word_index) + " "
    pattern = re.compile(r".+\s.+\s(.+)")
    return s[:len(s)-1], pattern.search(s).group(1)


def get_word_from_index(dependencies, index):
    for dep in dependencies:
        if dep['dependent'] == index:
            return dep['dependentGloss']


def get_start_end_indices(sentence, start_word_index, end_word_index, end_word):
    print("sentence", sentence)
    print("start", start_word_index)
    print("end", end_word_index)
    char_counter = 0
    word_counter = 0
    start = 0
    sentence = sentence.split(" ")
    for word in sentence:
        word_counter = word_counter + 1
        if word_counter == start_word_index:
            start = char_counter
        char_counter = char_counter + len(word) + 1
        if word_counter == end_word_index:
            return start, char_counter - 1


def main(data):
    pattern = re.compile(r"(evaluation-phase1-master/.+/.+)/.+Stanza-out.txt")
    pattern2 = re.compile(r"evaluation-phase1-master/(.+/.+)/.+Stanza-out.txt")
    for file in data:
        folder = pattern.search(file).group(1)
        folder2 = "evaluation-phase2-master/"+pattern2.search(file).group(1)
        sentences = ""
        entities = ""
        triples = {"1": [False, "(Contribution||has||Ablation Analysis)\n"],
                       "2": [False, "(Contribution||has||Approach)\n"],
                       "3": [False, "(Contribution||has||Baselines)\n"],
                       "4": [False, "(Contribution||has||Code)\n"],
                       "5": [False, "(Contribution||has||Dataset)\n"],
                       "6": [False, "(Contribution||has||Experimental Setup)\n"],
                       "7": [False, "(Contribution||has||Experiments)\n"],
                       "8": [False, "(Contribution||has||Hyperparameters)\n"],
                       "9": [False, "(Contribution||has||Model)\n"],
                       "11": [False, "(Contribution||has||Research Problem)\n"],
                       "12": [False, "(Contribution||has||Results)\n"],
                       "13": [False, "(Contribution||has||Tasks)\n"]
                       }

        for sentence_index in data[file]:
            info_unit = data[file][sentence_index][0]
            if info_unit == '10':
                info_unit = '9'
            elif info_unit == '0':
                continue
            sentence = data[file][sentence_index][1]
            labeled_sentence = data[file][sentence_index][2]
            dependencies = data[file][sentence_index][3]
            root = dependencies[0]['dependent']
            root_phrase = get_verb_phrase(dependencies, root)
            subject = get_root_subject(dependencies, root)
            subject_phrase = get_subject_phrase(dependencies, subject)
            if subject_phrase[0] == None:
                continue
            numbered_phrase, phrase = get_string_from_phrase(
                    dependencies, subject_phrase, sentence, sentence_index)
            if not phrase_in_entities(phrase, labeled_sentence):
                continue
            object = get_root_object(dependencies, root)
            object_phrase, root_phrase = get_object_phrase(dependencies, object, root_phrase)
            if object_phrase[0] == None:
                continue
            numbered_phrase, phrase = get_string_from_phrase(dependencies, object_phrase, sentence, sentence_index)
            if not phrase_in_entities(phrase, labeled_sentence):
                continue
            second_verb = get_secondary_verb(dependencies, subject)
            if subject_phrase[0] == None:
                sub = info_unit
            else:
                sub, sub_name = get_string_from_phrase(dependencies, subject_phrase, sentence, sentence_index)
            root, root_name = get_string_from_phrase(dependencies, root_phrase, sentence, sentence_index)
            obj, obj_name = get_string_from_phrase(dependencies, object_phrase, sentence, sentence_index)
            sentences = sentences + sentence_index+"\n"
            entities = entities + sub + "\n" + root + "\n" + obj + "\n"
            triples[info_unit][1] = triples[info_unit][1] + "(" + sub_name[:len(sub_name)-1] + "||" + root_name[:len(root_name)-1] + "||" + obj_name[:len(obj_name)-1] + ")\n"
            triples[info_unit][0] = True
            if second_verb is not None:
                verb_phrase = get_verb_phrase(dependencies, second_verb)
                second_object = get_root_object(dependencies, second_verb)
                second_object_phrase, verb_phrase = get_object_phrase(dependencies, second_object, verb_phrase)
                print("dependences", dependencies)
                print("second", second_object_phrase)
                print("sentence", sentence)
                print("index", sentence_index)
                if second_object_phrase[0] is None:
                    continue
                numbered_phrase, phrase = get_string_from_phrase(dependencies, second_object_phrase, sentence, sentence_index)
                if not phrase_in_entities(phrase, labeled_sentence):
                    continue
                #sub = get_string_from_phrase(dependencies, subject_phrase, sentence, sentence_index)
                vb, vb_name = get_string_from_phrase(dependencies, verb_phrase, sentence, sentence_index)
                obj2, obj2_name = get_string_from_phrase(dependencies, second_object_phrase, sentence, sentence_index)
                entities = entities + sub + "\n" + vb + "\n" + obj2 + "\n"
                triples[info_unit][1] = triples[info_unit][1] + "(" + sub_name[:len(sub_name)-1] + "||" + vb_name[:len(vb_name)-1] + "||" + obj2_name[:len(obj2_name)-1] + ")\n"
                triples[info_unit][0] = True
        print(sentences)
        print(entities)
        #with open("../"+folder+"/sentences.txt", "w") as f:
            #f.write(sentences)
        #f.close()
        print("../" + folder2 + "/entities.txt")
        with open("../"+folder2+"/entities.txt", "w") as f:
            f.write(entities)
        f.close()
        if not os.path.isdir("../"+folder2+"/triples/"):
            os.mkdir("../"+folder2+"/triples")
        else:
            shutil.rmtree("../"+folder2+"/triples")
            os.mkdir("../" + folder2 + "/triples")
        for triple in triples:
            if triples[triple][0]:
                with open("../"+folder2+"/triples/"+iu_dict[triple]+".txt", "w") as f:
                    f.write(triples[triple][1])
                f.close()
main(input_data)