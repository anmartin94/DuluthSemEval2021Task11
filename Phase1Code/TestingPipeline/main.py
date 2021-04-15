#from TestingPipeline import memm_ner, dep_parse, extract_info
#from TestingPipeline import classify_sentences, label_sentences, dep_parse, extract_info
import classify_sentences, label_sentences, dep_parse, extract_info
import re

with open("Phase1Code/DataFiles/sci-phrases.txt", "r") as f:
    words = list(f.readlines())
f.close()
for word in words:
    word = word[:len(word)-1]
data = {}
pattern = re.compile(r"(.+)\t(.+)\t(.+)\t(.+)")
with open("Phase1Code/DataFiles/output.txt", "r") as f:
    for line in f.readlines():
        line = line[:len(line)-1]
        parsed = pattern.search(line)
        if data.get(parsed.group(1)) is None:
            data[parsed.group(1)] = {parsed.group(2): [parsed.group(3), parsed.group(4)]}
        else:
            data[parsed.group(1)][parsed.group(2)] = [parsed.group(3), parsed.group(4)]
f.close()
print("Labeling scientific terms")
data = label_sentences.main(data)
for path in data:
    for index in data[path]:
        for labeled in data[path][index][2]:
            if labeled[1] != 'O':
                found = False
                #print(labeled[0])
                for word in words:
                    if labeled[0].lower() in word.lower():
                        found = True
                        break
               


print("Creating dependency parse")
data = dep_parse.main(data)
print("Extracting info into sentences, entities, and triples files")
extract_info.main(data)
