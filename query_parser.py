import xml.etree.ElementTree as ET;
import codecs;

class QueryParser:
    def __init__(self, XMLFileName):
        self._selectedTags = ["questions", "concepts"];
        self._tagWeights = {"questions": 1, "concepts": 4};
        self._queryTerms = {"questions": {}, "concepts": {}};
        
        self._queryCount = 0;
        self._queryData = {"numbers": [], "titles": [], "questions": [], "narratives": [], "concepts": []};

        self._readXMLFile(XMLFileName);

    def _readXMLFile(self, fileName):
        tree = ET.parse(fileName);
        root = tree.getroot();

        self._queryCount = len(root);

        for topic in root:
            for number in topic.findall("number"):
                self._queryData["numbers"].append(number.text[-3 : ]);

            for title in topic.findall("title"):
                self._queryData["titles"].append(title.text);

            for question in topic.findall("question"):
                self._queryData["questions"].append(question.text);

            for narrative in topic.findall("narrative"):
                self._queryData["narratives"].append(narrative.text);

            for concept in topic.findall("concepts"):
                self._queryData["concepts"].append(concept.text);

    def getQueryCount(self):
        return self._queryCount;

    def getQueryNumber(self, index):
        return self._queryData["numbers"][index];

    def getTermInTag(self, termID):
        highestWeightTag = None;

        for tag in self._selectedTags:
            if termID in self._queryTerms[tag] and (highestWeightTag == None or self._tagWeights[highestWeightTag] < self._tagWeights[tag]):
                highestWeightTag = tag;

        return highestWeightTag;

    def getTermWeight(self, termID):
        sumWeights = 0;

        for tag in self._selectedTags:
            if termID in self._queryTerms[tag]:
                sumWeights += self._tagWeights[tag];

        return sumWeights;

    def getTagWeight(self, tag):
        return self._tagWeights[tag];

    def getQueryVectors(self, dataset):
        queryVectors = [{} for i in xrange(self._queryCount)];

        for (q, queryVector) in enumerate(queryVectors):
            for tag in self._selectedTags:
                tagLength = len(self._queryData[tag][q]);

                for (t, term1) in enumerate(self._queryData[tag][q]):
                    termID1 = dataset.convertTermNameToID(term1);
                    if termID1 == None:                             # no such unigram term in the dataset
                        continue;
                    
                    if termID1 in queryVector:                      # unigram
                        queryVector[termID1] += 1;
                    else:
                        queryVector[termID1] = 1;

                    self._queryTerms[tag][termID1] = 1;
                    
                    if t < tagLength - 1:
                        term2 = self._queryData[tag][q][t + 1];
                        termID2 = dataset.convertTermNameToID(term2);
                        if termID2 == None or not dataset.hasTerm((termID1, termID2)):   # no such bigram term in the dataset
                            continue;

                        if (termID1, termID2) in queryVector:       # bigram
                            queryVector[(termID1, termID2)] += 2;
                        else:
                            queryVector[(termID1, termID2)] = 2;

                        self._queryTerms[tag][(termID1, termID2)] = 1;

        for termID in queryVector.keys():
            if self.getTermInTag(termID) != "concepts":
                del queryVector[termID];
            
        return queryVectors;
