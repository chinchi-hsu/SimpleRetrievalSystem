systemDirectory = /tmp2/r02922012/InformationRetrieval/
queryTrainFile = queries/query-train.xml
queryTestFile = queries/query-test.xml
rankListTrainFile = train.out
rankListTestFile = test.out
modelDirectory = wm/
datasetDirectory = CIRB010/
answerRankListFile = queries/ans-train
stopWordFile = stoplist.zh_TW.u8
feedback = ""

train:
	./execute.sh $(feedback) -i $(systemDirectory)$(queryTrainFile) -o $(rankListTrainFile) -m $(systemDirectory)$(modelDirectory) -d $(systemDirectory)$(datasetDirectory) -a $(systemDirectory)$(answerRankListFile) -s $(stopWordFile)

test:
	./execute.sh $(feedback) -i $(systemDirectory)$(queryTestFile) -o $(rankListTestFile) -m $(systemDirectory)$(modelDirectory) -d $(systemDirectory)$(datasetDirectory) -s $(stopWordFile)
