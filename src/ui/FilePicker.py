# %matplotlib tk
from Tkinter import Tk
from tkFileDialog import askopenfilename, askdirectory
dirPickTitle = "Select training directory"

#instantiate a Tk window
root = Tk()

#set the title of the window
root.title(dirPickTitle)
root.update()

# Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
# show an "Open" dialog box and return the path to the selected file
directory = askdirectory(initialdir = "/",
                         title = dirPickTitle)
print(directory)