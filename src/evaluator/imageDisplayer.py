import Tkinter as tk
from Tkinter import *
from PIL import Image,ImageTk
import os,os.path


def callBackLike():
    print 'like'


def callBackDislike():
    print 'in dislike'



path='images/'
temp=[]
for f in os.listdir(path):
    temp.append(Image.open(os.path.join(path, f)))

for image in temp:
    root=Tk()
    img=ImageTk.PhotoImage(image)
    panel=tk.Label(root,image=img)
    panel.pack(side="top")
    b1= Button(root, text="Like", command=callBackLike)
    b1.pack()
    b2=Button(root, text='Dislike', command=callBackDislike)
    b2.pack()
    root.mainloop()



