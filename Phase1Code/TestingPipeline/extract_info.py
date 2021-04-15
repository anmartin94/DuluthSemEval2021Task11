import os
import re
import shutil

from nltk import word_tokenize

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
    if not os.path.isdir("submission/"):
        os.mkdir("submission",mode = 0o777)
    else:
        shutil.rmtree("submission")
        os.mkdir("submission",mode = 0o777)
    os.chdir("submission")
    pattern = re.compile(r"(evaluation-phase1-master/(.+)/(.+))/.+Stanza-out.txt")
    #pattern2 = re.compile(r"evaluation-phase2-master/(.+/.+)/.+Stanza-out.txt")
    for file in data:
        searched = pattern.search(file)
        folder = searched.group(1)
        named_folder = searched.group(2)
        numbered_folder = searched.group(3)
        folder = folder.replace("evaluation-phase1-master", "submission")
        if not os.path.isdir(named_folder):
            os.mkdir(named_folder,mode = 0o777)
        os.chdir(named_folder)
        os.mkdir(numbered_folder,mode = 0o777)
        os.chdir(numbered_folder)
        #folder2 = "evaluation-phase2-master/"+pattern2.search(file).group(1)
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
        with open("sentences.txt", "w") as f:
            f.write(sentences)
        f.close()
        with open("entities.txt", "w") as f:
            f.write(entities)
        f.close()
        if not os.path.isdir("triples/"):
            os.mkdir("triples")
        else:
            shutil.rmtree("triples")
            os.mkdir("triples")
        for triple in triples:
            if triples[triple][0]:
                with open("triples/"+iu_dict[triple]+".txt", "w") as f:
                    f.write(triples[triple][1])
                f.close()
        os.chdir("../..")

