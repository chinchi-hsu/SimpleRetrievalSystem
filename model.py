import math;

class VectorSpaceModel:
    def __init__(self, queryVector, dataset, queryParser):
        self._queryVector = queryVector;
        self._dataset = dataset;
        self._queryParser = queryParser;

        #self._reweightWFIDF(self._queryVector);
        #self._queryVectorNorm2 = self._getNorm2(self._queryVector);

    def _getNorm2(self, vector):
        return math.sqrt(sum([element * element for (dimension, element) in vector.items()]));

    def _getCosineSimilarity(self, queryVector, documentVector):
        innerProduct = 0.0;

        #for termID in queryVector.keys():
        #    if termID in documentVector:
        #        innerProduct += queryVector[termID] * documentVector[termID];
        for termID in documentVector.keys():                            # document vector is extracted from query vector => its dimension is sparser
            innerProduct += queryVector[termID] * documentVector[termID];
        
        return innerProduct / (self._queryVectorNorm2 * self._getNorm2(documentVector));
    
    def _getOkapiSimilarity(self, queryVector, documentVector, documentID, relevantDocumentFrequencies, irrelevantDocumentFrequencies, relevantDocumentCount):
        k1 = 1.6;
        k3 = 1.8;
        b = 0.75;
        LRatio = float(self._dataset.getDocumentLength(documentID)) / self._dataset.getAverageDocumentLength();
        similarity = 0.0;
        
        for termID in queryVector.keys():
            if termID in documentVector:
                if relevantDocumentFrequencies:
                    documentFrequency = self._dataset.getDocumentFrequency(termID);
                    documentCount = self._dataset.getDocumentCount();

                    IDFTerm1 = relevantDocumentFrequencies[termID] + 0.5;
                    IDFTerm2 = irrelevantDocumentFrequencies[termID] + 0.5;
                    IDFTerm3 = documentFrequency - relevantDocumentFrequencies[termID] + 0.5;
                    IDFTerm4 = documentCount - documentFrequency - relevantDocumentCount + relevantDocumentFrequencies[termID] + 0.5;
                    IDF = (IDFTerm1 / IDFTerm2) / (IDFTerm3 / IDFTerm4);
                    documentTF = float((k1 + 1) * documentVector[termID]) / (k1 * ((1 - b) + b * LRatio) + documentVector[termID]);
                    queryTF = float((k3 + 1) * queryVector[termID]) / (k3 + queryVector[termID]);
                    termTag = self._queryParser.getTermInTag(termID);

                    similarity += math.log10(IDF * documentTF * queryTF);
                    
                else:
                    IDF = self._dataset.getTermInverseDocumentFrequency(termID);
                    documentTF = float((k1 + 1) * documentVector[termID]) / (k1 * ((1 - b) + b * LRatio) + documentVector[termID]);
                    queryTF = float((k3 + 1) * queryVector[termID]) / (k3 + queryVector[termID]);
                    termTag = self._queryParser.getTermInTag(termID);

                    similarity += IDF * documentTF * queryTF * self._queryParser.getTagWeight(termTag);

        return similarity;

    def _reweightWFIDF(self, vector):
        for termID in vector.keys():
            vector[termID] = 1 + math.log10(vector[termID]) if vector[termID] > 0 else 0;
            vector[termID] *= self._dataset.getTermInverseDocumentFrequency(termID);

    def _getFeedbackDocumentFrequencies(self, relevantDocuments, irrelevantDocuments):
        relevantDocumentFrequencies = {};
        irrelevantDocumentFrequencies = {};

        for termID in self._queryVector.keys():
            relevantDocumentFrequencies[termID] = 0;
            irrelevantDocumentFrequencies[termID] = 0;

            if relevantDocuments:
                for documentID in relevantDocuments:
                    if self._dataset.documentHasTerm(documentID, termID):
                        relevantDocumentFrequencies[termID] += 1;

            if irrelevantDocuments:
                for documentID in irrelevantDocuments:
                    if self._dataset.documentHasTerm(documentID, termID):
                        irrelevantDocumentFrequencies[termID] += 1;

        return (relevantDocumentFrequencies, irrelevantDocumentFrequencies);

    def getRankList(self, topRankK, relevantDocuments, irrelevantDocuments):
        rankDocuments = {};
        candidateDocuments = self._dataset.getRelatedDocuments(self._queryVector);
        print("\t\tNumber of candidate documents: %d" % (len(candidateDocuments)));

        relevantDocumentFrequencies = {};
        irrelevantDocumentFrequencies = {};
        relevantDocumentCount = 0;
        if relevantDocuments:
            (relevantDocumentFrequencies, irrelevantDocumentFrequencies) = self._getFeedbackDocumentFrequencies(relevantDocuments, irrelevantDocuments);
            relevantDocumentCount = len(relevantDocuments);

        for documentID in candidateDocuments:
            documentVector = self._dataset.getDocumentVector(documentID, self._queryVector);
            #self._reweightWFIDF(documentVector);
            #rankDocuments[documentID] = self._getCosineSimilarity(self._queryVector, documentVector);
            rankDocuments[documentID] = self._getOkapiSimilarity(self._queryVector, documentVector, documentID, relevantDocumentFrequencies, irrelevantDocumentFrequencies, relevantDocumentCount);

        sortedRankDocuments = sorted(rankDocuments.items(), key = lambda (k, v): (v, k), reverse = True);
        retrievedDocuments = [];
        for (n, (documentID, similarity)) in enumerate(sortedRankDocuments):
            if n >= topRankK:
                break;
            retrievedDocuments.append(documentID);

        return retrievedDocuments;

class RacchioRelevanceFeedback:
    def __init__(self, alpha, beta, gamma):
        self._alpha = alpha;
        self._beta = beta;
        self._gamma = gamma;

    def updateQueryVector(self, queryVector, dataset, rankList, topRankK, bottomRankK):
        averagePositiveVector = {};
        averageNegativeVector = {};

        for d in xrange(topRankK):
            documentID = rankList[d];
            documentVector = dataset.getDocumentVector(documentID, queryVector);
            for termID in documentVector.keys():
                if termID in averagePositiveVector:
                    averagePositiveVector[termID] += documentVector[termID];
                else:
                    averagePositiveVector[termID] = documentVector[termID];

        for termID in averagePositiveVector.keys():
            averagePositiveVector[termID] /= float(topRankK);

        for d in xrange(len(rankList) - bottomRankK, len(rankList)):
            documentID = rankList[d];
            documentVector = dataset.getDocumentVector(documentID, queryVector);
            for termID in documentVector.keys():
                if termID in averageNegativeVector:
                    averageNegativeVector[termID] += documentVector[termID];
                else:
                    averageNegativeVector[termID] = documentVector[termID];

        for termID in averageNegativeVector.keys():
            averageNegativeVector[termID] /= float(bottomRankK);

        for termID in queryVector.keys():
            queryVector[termID] = self._alpha * queryVector[termID];
            
            if termID in averagePositiveVector:
                queryVector[termID] += self._beta * averagePositiveVector[termID];

            if termID in averageNegativeVector:
                queryVector[termID] -= self._gamma * averageNegativeVector[termID];

        return queryVector;

