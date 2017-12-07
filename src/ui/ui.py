import matplotlib#needed to make sure TKinter doesn't crash!
matplotlib.use('TkAgg') #needed to make sure TKinter doesn't crash!
from Tkinter import Tk, Label, Entry
import sys
from PIL import Image,ImageTk,ImageFilter,ImageOps
class UI:

	def __init__(self):
		self.root = Tk()
		lbl_pickTrainingDir = Label(self.root, text="Choose Training File").grid(row=1, column=0)
		txt_pickedTrainingDir = Entry(self.root).grid(row=1, column=1)
		lbl_pickTestingDir = Label(self.root, text = "Choose Directory with images to pass to agent").grid(row = 2, column = 0)
		txt_pickedTestingDir = Entry(self.root).grid(row = 2, column = 1)