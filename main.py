from query_parser import QueryParser;
from dataset import Dataset;
from model import VectorSpaceModel;
from model import RacchioRelevanceFeedback;
from evaluation import Evaluation;
import sys;
import copy;

class RetrievalSystem:
    def __init__(self, invertedIndexFileName, vocabularyFileName, fileListFileName, stopWordFileName, queryFileName, answerRankListFileName, relevanceFeedbackSwitch):

        print("\tRead input data");
        self._dataset = Dataset(invertedIndexFileName, vocabularyFileName, fileListFileName, stopWordFileName);

        print("\tRead XML query file");
        self._queryData = QueryParser(queryFileName);

        print("\tExtract query vectors from query file");
        self._queryVectors = self._queryData.getQueryVectors(self._dataset);
        
        self._rankLists = [];
        self._bestRankLists = [];
        self._relevanceFeedback = relevanceFeedbackSwitch;

        if answerRankListFileName:
            self._trainMode = True;
            self._answerRankLists = self._readAnswerRankLists(answerRankListFileName);
        else:
            self._trainMode = False; # i.e. test mode

    def _readAnswerRankLists(self, answerRankListFileName):
        answerRankLists = [[] for q in xrange(len(self._queryVectors))];
        inFile = open(answerRankListFileName, "r");

        for line in inFile:
            item = line.split();
            index = int(item[0]) - 1;
            documentID = self._dataset.convertDocumentNameToID(item[1]);
            
            answerRankLists[index].append(documentID);

        inFile.close();
        return answerRankLists;

    def retrieve(self, topRankK, relevantDocumentList, irrelevantDocumentList):
        self._rankLists = [];

        for (q, queryVector) in enumerate(self._queryVectors):
            print("\tSearch for the query %d" % (q));
            vectorSpaceModel = VectorSpaceModel(queryVector, self._dataset, self._queryData);
            rankList = vectorSpaceModel.getRankList(100, relevantDocumentList[q] if relevantDocumentList else None, irrelevantDocumentList[q] if irrelevantDocumentList else None);
            self._rankLists.append(rankList);

        if self._trainMode:
            evaluation = Evaluation(self._rankLists, self._answerRankLists);
            MAP = evaluation.evaluateMeanAveragePrecision(topRankK);
            print("\tMAP: %f" % (MAP));
            return MAP;
        else:
            return 0;
        
    def outputPredictions(self, fileName, topRankK):
        outFile = open(fileName, "w");

        for (q, rankList) in enumerate(self._bestRankLists):
            for (d, documentID) in enumerate(rankList):
                if d >= topRankK:
                    break;
                document = self._dataset.convertDocumentIDToName(documentID);
                outFile.write("%s %s%s\n" % (self._queryData.getQueryNumber(q), document, " *" if self._trainMode and (documentID in self._answerRankLists[q]) else ""));

        outFile.close();

    def runFeedbackQuerySearch(self, iterations):
        maxMAPIteration = 0;
        maxMAP = self.retrieve(100, None, None);
        self._bestRankLists = copy.deepcopy(self._rankLists);
        print("\tMAP: %f" % (maxMAP));
        for n in xrange(1, iterations + 1):
            print("\tIteration %d: Update rank list" % (n));
            relevantDocumentList = [];
            irrelevantDocumentList = [];

            for rankList in self._rankLists:
                relevantDocumentList.append(rankList[0 : 10]);      # Top 10 as relevant documents
            
            MAP = self.retrieve(100, relevantDocumentList, irrelevantDocumentList);
            if self._trainMode:
                print("\tMAP: %f" % (MAP));
                if maxMAP < MAP:
                    maxMAP = MAP;
                    maxMAPIteration = n;
                    self._bestRankLists = copy.deepcopy(self._rankLists);
            else:
                self._bestRankLists = copy.deepcopy(self._rankLists);

        if self._trainMode:
            print("Max MAP: %f, iteration: %d" % (maxMAP, maxMAPIteration));
            return (maxMAP, maxMAPIteration);

    def run(self):
        if self._relevanceFeedback:
            self.runFeedbackQuerySearch(10);
        else:
            self.runFeedbackQuerySearch(0);

def main():
    queryFileName = "";
    rankListFileName = "";
    modelDirectoryName = "";
    datasetDirectoryName = "";
    answerRankListFileName = "";
    stopWordFileName = "";
    relevanceFeedbackSwitch = False;

    a = 1;
    argumentCount = len(sys.argv);
    print("Number of arguments: %d" % (argumentCount));
    while a < argumentCount:
        if sys.argv[a][0] != "-":
            print "Command format incorrect!";
            quit();

        command = sys.argv[a][1];

        if command == "r":
            relevanceFeedbackSwitch = True;

        elif command == "i":
            a += 1;
            queryFileName = sys.argv[a];

        elif command == "o":
            a += 1;
            rankListFileName = sys.argv[a];

        elif command == "m":
            a += 1;
            modelDirectoryName = sys.argv[a];

        elif command == "d":
            a += 1;
            datasetDirectoryName = sys.argv[a];

        elif command == "a":
            a += 1;
            answerRankListFileName = sys.argv[a];

        elif command == "s":
            a += 1;
            stopWordFileName = sys.argv[a];

        a += 1;

    if modelDirectoryName[-1] == "/":
        modelDirectoryName = modelDirectoryName[ : -1]; # removes the slash in the end if it exists
    invertedIndexFileName = modelDirectoryName + "/inverted-index";
    vocabularyFileName = modelDirectoryName + "/vocab.all";
    fileListFileName = modelDirectoryName + "/file-list";
   
    print("========");
    print("Show file paths");
    print("\tInverted index file: %s" % (invertedIndexFileName));
    print("\tVocabulary file: %s" % (vocabularyFileName));
    print("\tFile list file: %s" % (fileListFileName));
    print("\tQuery file: %s" % (queryFileName));
    print("\tRank list file: %s" % (rankListFileName));
    print("\tAnswer rank list file: %s" % (answerRankListFileName));
    print("\tStop word file: %s" % (stopWordFileName));
    
    print("========");

    print("Construct the retrieval system");
    retrievalSystem = RetrievalSystem(invertedIndexFileName, vocabularyFileName, fileListFileName, stopWordFileName, queryFileName, answerRankListFileName, relevanceFeedbackSwitch);
    
    print("========");
    
    print("Run the retrieval system");
    retrievalSystem.run();
    
    print("========");
    
    print("Output the predictions from the retrieval system")
    retrievalSystem.outputPredictions(rankListFileName, 100);

if __name__ == "__main__":
    main();
