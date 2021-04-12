'''This file uses the Stanford Core NLP library to annotate every sentence with its dependencies.
Uses a Stanford CoreNLP Python wrapper written by Smitha Milli https://github.com/smilli/py-corenlp
To be able to run this, you must download CoreNLP 4.2.0 https://stanfordnlp.github.io/CoreNLP/download.html
After downloading, run the following line from the directory you downloaded it to:
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize,ssplit,pos,lemma,parse,sentiment" -port 9000 -timeout 30000
'''

import re
from pycorenlp import StanfordCoreNLP


input_data = {"../evaluation-phase1-master/data-to-text_generation/4/1809.00582v2-Stanza-out.txt" :{
    '1': ['0', 'title', [['title', 'O']]],
    '2': ['11', 'Data - to - Text Generation with Content Selection and Planning', [['Data', 'O'], ['to', 'O'], ['Text', 'O'], ['Generation', 'O'], ['with', 'O'], ['Content', 'O'], ['Selection', 'O'], ['and', 'O'], ['Planning', 'B']]],
    '3': ['0', 'abstract', [['abstract', 'O']]],
    '4': ['0', 'Recent advances in data - to - text generation have led to the use of large - scale datasets and neural network models which are trained end - to - end , without explicitly modeling what to say and in what order .', [['Recent', 'O'], ['advances', 'O'], ['in', 'O'], ['data', 'O'], ['to', 'O'], ['text', 'O'], ['generation', 'O'], ['have', 'O'], ['led', 'O'], ['to', 'O'], ['the', 'O'], ['use', 'O'], ['of', 'O'], ['large', 'B'], ['scale', 'O'], ['datasets', 'O'], ['and', 'O'], ['neural', 'B'], ['network', 'B'], ['models', 'O'], ['which', 'O'], ['are', 'O'], ['trained', 'O'], ['end', 'O'], ['to', 'O'], ['end', 'O'], ['without', 'O'], ['explicitly', 'O'], ['modeling', 'O'], ['what', 'O'], ['to', 'O'], ['say', 'O'], ['and', 'O'], ['in', 'O'], ['what', 'O'], ['order', 'O']]]}}


def main(input_data):
    for data in input_data:
        for id in input_data[data]:
            if input_data[data][id][0] == "0":
                continue
            sentence = input_data[data][id][1]
            no_percent = re.sub(r"(-|%)", '', sentence)
            nlp_wrapper = StanfordCoreNLP("http://localhost:9000")
            annotation = nlp_wrapper.annotate(no_percent,
                                              properties={
                                                  'annotators': 'depparse',
                                                  'outputFormat': 'json',
                                                  'timeout': 10000})
            dependencies = annotation["sentences"][0]["basicDependencies"]
            input_data[data][id].append(dependencies)
    return input_data




