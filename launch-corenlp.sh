#!/bin/bash
corenlp() {
        cd stanford-corenlp-4.2.0/
        var=$(pwd)
        echo $var
        java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -annotators "tokenize,ssplit,pos,lemma,parse,sentiment" -port 9000 -timeout 30000
}

corenlp
