import Tkinter as tk
from Tkinter import *
from PIL import Image,ImageTk
import os,os.path
import shutil
import datetime, time

class Evaluator:

	def __init__(self, trainingDirectory=os.path.abspath('../data/training/ling'), evaluationDirectory=os.path.abspath('../data/enhance')):
		self.evaluationDirectory = evaluationDirectory
		self.trainingDirectory = trainingDirectory
		print self.evaluationDirectory, self.trainingDirectory

	def __getFilesToBeDisplayed__(self):
		'''

		:return: List of images in evaluation directory
		'''

		fileList = []
		for f in os.listdir(self.evaluationDirectory):
			if f.lower().endswith('jpg') or f.lower().endswith('jpeg') or f.lower().endswith('png'):
				fileList.append(os.path.join(self.evaluationDirectory, f))

		return fileList


	def showImages(self):
		'''

		:return: Show images in training directory
		'''

		imageFileList = self.__getFilesToBeDisplayed__()
		for imageFile in imageFileList:
			root = Tk()
			resizedImage = Image.open(imageFile)
			resizedImage = resizedImage.resize((500, 500), Image.ANTIALIAS)  # The (250, 250) is (height, width)
			tkinterImg = ImageTk.PhotoImage(resizedImage)
			panel = tk.Label(root, image = tkinterImg)
			panel.pack(side = "top")
			b1 = Button(root, text = "Like", command = lambda: likeDislikeCallback(1, imageFile, root, self))
			b1.pack()
			b2 = Button(root, text = 'Dislike', command = lambda: likeDislikeCallback(1, imageFile, root, self))
			b2.pack()
			root.mainloop()

	@staticmethod
	def resizeImage(imageFile):
		resizedImage = Image.open(imageFile)
		resizedImage = resizedImage.resize((500, 500), Image.ANTIALIAS)
		return resizedImage

	def moveImageToTraining(self, imageFile):
		fileName = os.path.basename(imageFile)
		destinationFilename = Evaluator.getTimestamp()+fileName
		destination = os.path.join(self.trainingDirectory, '1',destinationFilename)
		print 'Moved %s to %s'%(imageFile, destination)
		shutil.copy(imageFile,destination)

	@staticmethod
	def getTimestamp():
		timestamp = datetime.datetime.fromtimestamp(time.time()  ).strftime('%Y%m%d%H%M%S')
		return timestamp


def likeDislikeCallback(choice, imageFile, tkRoot, evaluator):
	LIKE = 1
	DISLIKE = 0
	tkRoot.destroy()
	if choice == LIKE:
		evaluator.moveImageToTraining(imageFile)
	else:
		print 'Dislike'


if __name__ == '__main__':
	Evaluator().showImages()