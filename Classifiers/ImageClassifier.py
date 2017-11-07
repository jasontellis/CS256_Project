from Constants import Constants
class ImageClassifier:

    def __init__(self, trainedModel):
        self.trainedModel = trainedModel

    def classify(self, imageVector):

        classLabel = self.trainedModel.predict(imageVector)
        return classLabel

ic = ImageClassifier(1)
ic.classify([0])
