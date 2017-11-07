class ImageClassifier:

    def __init__(self, trainedModel):
        self.trainedModel = trainedModel

    def classify(self, imageVector):

        classLabel = self.trainedModel.predict(imageVector)
        return classLabel
