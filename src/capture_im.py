import tkinter as tk
from tkinter import *
import os
import cv2
import sys
from PIL import Image, ImageTk
import numpy

fileName = 'C:\\Users\\Paramveer\\Desktop\\Images1'+'\\WebcamCap.txt'
cancel = False
filepath = 'C:\\Users\\Paramveer\\Desktop\\Images1' 

count=0

def prompt_ok(event = 0):
    global cancel, button, button1, button2, entry1, entry2, entry3, entry4
    cancel = True

    button.place_forget()
    entry1 = Entry(mainWindow, textvariable=StringVar)
    label1 = Label(mainWindow, text="Name")
    entry2 = Entry(mainWindow, textvariable=StringVar)
    label2 = Label(mainWindow, text="Gender")
    entry3 = Entry(mainWindow, textvariable=StringVar)
    label3 = Label(mainWindow, text="City")
    entry4 = Entry(mainWindow, textvariable=StringVar)
    label4 = Label(mainWindow, text="Age")
    button1 = tk.Button(mainWindow, text="Save Image", command=save)
    button2 = tk.Button(mainWindow, text="Next", command=resume)
    button1.place(anchor=tk.CENTER, relx=0.2, rely=0.9, width=150, height=50)
    button2.place(anchor=tk.CENTER, relx=0.8, rely=0.9, width=150, height=50)
    entry1.place(anchor=tk.CENTER, relx=0.15, rely=0.1, width=190, height=25)
    label1.place(anchor=tk.CENTER, relx=0.35, rely=0.1, width=50, height=25)
    entry2.place(anchor=tk.CENTER, relx=0.55, rely=0.1, width=150, height=25)
    label2.place(anchor=tk.CENTER, relx=0.72, rely=0.1, width=50, height=25)
    entry3.place(anchor=tk.CENTER, relx=0.15, rely=0.2, width=190, height=25)
    label3.place(anchor=tk.CENTER, relx=0.35, rely=0.2, width=50, height=25)
    entry4.place(anchor=tk.CENTER, relx=0.55, rely=0.2, width=150, height=25)
    label4.place(anchor=tk.CENTER, relx=0.72, rely=0.2, width=50, height=25)
    button1.focus()         

def resume(event = 0):
    global button1, button2, button, lmain, cancel

    cancel = False

    button1.place_forget()
    button2.place_forget()

    mainWindow.bind('<Return>', prompt_ok)
    button.place(bordermode=tk.INSIDE, relx=0.5, rely=0.9, anchor=tk.CENTER, width=300, height=50)
    lmain.after(10, show_frame)
    
def save(event = 0):
    global prevImg, count
    
    if not os.path.exists(filepath+'\\'+entry1.get()):    
        os.mkdir(filepath+'\\'+entry1.get())
        fh = open(filepath+'\\'+entry1.get()+'\\meta.csv', 'w')  
        fh.write(entry1.get()+','+entry2.get()+','+entry3.get()+','+entry4.get()+'\n') 
        fh.close()

    prevImg.save(filepath+'\\'+entry1.get()+'\\image'+str(count)+'.jpg')
    count=count+1

    resume()



def changeCam(event=0, nextCam=-1):
    global camIndex, cap, fileName

    if nextCam == -1:
        camIndex += 1
    else:
        camIndex = nextCam
    del(cap)
    cap = cv2.VideoCapture(camIndex)

    #try to get a frame, if it returns nothing
    success, frame = cap.read()
    if not success:
        camIndex = 0
        del(cap)
        cap = cv2.VideoCapture(camIndex)

    f = open(fileName, 'w')
    f.write(str(camIndex))
    f.close()

try:
    f = open(fileName, 'r')
    camIndex = int(f.readline())
except:
    camIndex = 0

cap = cv2.VideoCapture(camIndex)
capWidth = cap.get(3)
capHeight = cap.get(4)

success, frame = cap.read()
if not success:
    if camIndex == 0:
        print("Error, No webcam found!")
        sys.exit(1)
    else:
        changeCam(nextCam=0)
        success, frame = cap.read()
        if not success:
            print("Error, No webcam found!")
            sys.exit(1)


mainWindow = tk.Tk(screenName="Camera Capture")
mainWindow.resizable(width=False, height=False)
mainWindow.bind('<Escape>', lambda e: mainWindow.quit())
lmain = tk.Label(mainWindow, compound=tk.CENTER, anchor=tk.CENTER, relief=tk.RAISED)
button = tk.Button(mainWindow, text="Capture", command=prompt_ok)
button_changeCam = tk.Button(mainWindow, text="Cam", command=changeCam)

lmain.pack()
button.place(bordermode=tk.INSIDE, relx=0.5, rely=0.9, anchor=tk.CENTER, width=300, height=50)
button.focus()
button_changeCam.place(bordermode=tk.INSIDE, relx=0.95, rely=0.1, anchor=tk.CENTER, width=50, height=50)

def show_frame():
    global cancel, prevImg, button

    _, frame = cap.read()
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    prevImg = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=prevImg)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    if not cancel:
        lmain.after(10, show_frame)

show_frame()
mainWindow.mainloop()