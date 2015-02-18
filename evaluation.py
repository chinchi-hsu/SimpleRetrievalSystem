class Evaluation:
    def __init__(self, predictionRankLists, answerRankLists):
        self._predictionRankLists = predictionRankLists;
        self._answerRankLists = answerRankLists;

    def evaluateMeanAveragePrecision(self, topRankK):
        meanAveragePrecision = 0.0;

        for (q, predictionRankList) in enumerate(self._predictionRankLists):
            averagePrecision = 0.0;
            trueRelevanceCount = 0;

            for (r, documentID) in enumerate(predictionRankList):
                if r >= topRankK:
                    break;
                if documentID in self._answerRankLists[q]:
                    rank = r + 1;
                    trueRelevanceCount += 1;
                    averagePrecision += float(trueRelevanceCount) / rank;

            averagePrecision /= len(self._answerRankLists[q]);
            meanAveragePrecision += averagePrecision;
            print("\t\tAverage precision for the query %d: %f" % (q, averagePrecision));

        meanAveragePrecision /= len(self._predictionRankLists);
        return meanAveragePrecision;
