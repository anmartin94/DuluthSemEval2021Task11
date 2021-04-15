'''This file uses the Stanford Core NLP library to annotate every sentence with its dependencies.
Uses a Stanford CoreNLP Python wrapper written by Smitha Milli https://github.com/smilli/py-corenlp
To be able to run this, you must download CoreNLP 4.2.0 https://stanfordnlp.github.io/CoreNLP/download.html
After downloading, run the following line from the directory you downloaded it to:
java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize,ssplit,pos,lemma,parse,sentiment" -port 9000 -timeout 30000
'''

import re
from pycorenlp import StanfordCoreNLP



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




