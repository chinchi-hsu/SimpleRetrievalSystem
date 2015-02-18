import math;
import codecs;
import sys;
import copy;

class Dataset:
    def __init__(self, invertedIndexFileName, vocabularyFileName, fileListFileName, stopWordFileName):
        self._invertedIndex = {};
        self._documentFrequency = {};
        self._termInverseDocumentFrequency = {};
        self._termToNumber = {};
        self._numberToTerm = {};
        self._fileToNumber = {};
        self._numberToFile = {};
        self._stopWordList = {};
        self._documentCount = 0;
        self._documentVectors = {};
        self._documentLengths = {};
        self._averageDocumentLength = 0;
        
        print("\t\tRead vocabulary file");
        self._readVocabularyFile(vocabularyFileName);
        print("\t\tRead stop word file");
        self._readStopWordFile(stopWordFileName);
        print("\t\tRead file list file");
        self._readFileListFile(fileListFileName);
        print("\t\tRead invertedIndex file");
        self._readInvertedIndexFile(invertedIndexFileName);
        print("\t\tConstruct IDF, document information");
        self._constructTermInverseDocumentFrequencies();
        self._constructDocumentVectors();
        self._constructDocumentLengths();

        print("\t\tNumber of document: %d" % (self._documentCount));

    def _readInvertedIndexFile(self, fileName):
        inFile = open(fileName, "r");

        while 1:
            line = inFile.readline();
            if not line:
                break;
            grams = line.split();
            gram1 = int(grams[0]);
            gram2 = int(grams[1]);
            documentFrequency = int(grams[2]);

            if documentFrequency == 0: # no such term in the dataset.
                continue;
            if self.convertTermIDToName(gram1) == None or (gram2 > -1 and self.convertTermIDToName(gram2) == None) or gram1 in self._stopWordList or (gram2 > -1 and gram2 in self._stopWordList):   # no such terms in vocabulary list
                for d in xrange(documentFrequency):
                    inFile.readline();
                continue;

            termID = gram1 if gram2 == -1 else (gram1, gram2);
            self._invertedIndex[termID] = {};
            self._documentFrequency[termID] = documentFrequency;

            for d in xrange(documentFrequency):
                documentInformation = inFile.readline().split();
                documentID = int(documentInformation[0]);
                termFrequency = int(documentInformation[1]);

                self._invertedIndex[termID][documentID] = termFrequency;
        
        inFile.close();

    def _readVocabularyFile(self, fileName):
        inFile = codecs.open(fileName, "r", "utf-8");
        
        for (lineNumber, line) in enumerate(inFile): 
            term = line.split("\n")[0];
            if lineNumber == 0:
                continue;
            if len(term) > 1:   # Non-Chinese terms usually has more than 1 characters, and hence we remove non-Chinese using length detection
                continue;

            self._termToNumber[term] = lineNumber;
            self._numberToTerm[lineNumber] = term;

        inFile.close();

    def _readFileListFile(self, fileName):
        inFile = open(fileName, "r");

        for (lineNumber, line) in enumerate(inFile):
            listFile = line.split()[0];
            lastSlashIndex = listFile.rfind("/"); # removes the path of directories
            listFile = listFile[lastSlashIndex + 1 : ].lower(); # grab only the lower-case file name
            self._numberToFile[lineNumber] = listFile;
            self._fileToNumber[listFile] = lineNumber;

        inFile.close();

        self._documentCount = len(self._numberToFile);

    def _readStopWordFile(self, fileName):
        inFile = codecs.open(fileName, "r", "utf-8");

        for line in inFile:
            stopWord = line.split()[0];
            number = self._termToNumber[stopWord];
            self._stopWordList[number] = 1;

        inFile.close();
    
    def _constructTermInverseDocumentFrequencies(self):
        for termID in self._invertedIndex.keys():
            self._termInverseDocumentFrequency[termID] = math.log10(float(self._documentCount) / self._documentFrequency[termID]);
            #print("%f" % (self._termInverseDocumentFrequency[termID]));
    
    def _constructDocumentVectors(self):
        for (termID, documentList) in self._invertedIndex.items():
            for (documentID, frequency) in documentList.items():
                if documentID not in self._documentVectors:
                    self._documentVectors[documentID] = {};
                self._documentVectors[documentID][termID] = frequency;

    def _constructDocumentLengths(self):
        self._averageDocumentLength = 0.0;

        for (documentID, documentVector) in self._documentVectors.items():
            sumFrequency = 0;

            for (termID, frequency) in documentVector.items():
                if isinstance(termID, tuple):               # if the term is a bigram
                    sumFrequency += frequency;

            self._documentLengths[documentID] = sumFrequency;
            self._averageDocumentLength += sumFrequency;

        self._averageDocumentLength /= self._documentCount;
        print("\t\tAverage document length: %f" % (self._averageDocumentLength));

    def getRelatedDocuments(self, queryVector):
        documents = {};
        threshold = self._documentCount / 10.0;

        for termID in queryVector.keys():
            if termID in self._invertedIndex:
                if self._documentFrequency[termID] > threshold: # removes terms appearing in too many documents
                    continue;

                for documentID in self._invertedIndex[termID].keys():
                    documents[documentID] = 1; # uses the non-repetition property of dictionary keys to avoid repetition of document IDs

        return documents.keys();

    def getDocumentVector(self, documentID, queryVector):
        if queryVector == None:
            return copy.deepcopy(self._documentVectors[documentID]);
        
        documentVector = {};

        for termID in queryVector.keys():
            if termID in self._documentVectors[documentID]:
                documentVector[termID] = self._documentVectors[documentID][termID];

        return documentVector;

    def getTermInverseDocumentFrequency(self, termID):
        return self._termInverseDocumentFrequency[termID] if termID in self._termInverseDocumentFrequency else None;

    def convertDocumentIDToName(self, documentID):
        return self._numberToFile[documentID] if documentID in self._numberToFile else None;

    def convertDocumentNameToID(self, documentName):
        return self._fileToNumber[documentName] if documentName in self._fileToNumber else None;

    def convertTermIDToName(self, termID):
        return self._numberToTerm[termID] if termID in self._numberToTerm else None;

    def convertTermNameToID(self, termName):
        return self._termToNumber[termName] if termName in self._termToNumber else None;

    def hasTerm(self, termID):
        return True if termID in self._invertedIndex else False;

    def getAverageDocumentLength(self):
        return self._averageDocumentLength;

    def getDocumentLength(self, documentID):
        return self._documentLengths[documentID] if documentID in self._documentLengths else None;

    def documentHasTerm(self, documentID, termID):
        return True if termID in self._documentVectors[documentID] else False;

    def getDocumentFrequency(self, termID):
        return self._documentFrequency[termID] if termID in self._documentFrequency else None;

    def getDocumentCount(self):
        return self._documentCount;

if __name__ == "__main__":
   fileName = sys.argv[1];

