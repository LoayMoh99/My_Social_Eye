# pip install Pillow
# pip install gTTS
# pip install pyttsx3
# pip install pywin32
# pip install opencv-python
# pip install keyboard
from argparse import Action
from logging import root
from tkinter import *
import tkinter as tk
from tkinter.tix import IMAGE
import PIL
from PIL import ImageTk, Image
from gtts import gTTS
import pyttsx3
import os
import cv2
import numpy as np
import keyboard
from main_camera_video_thread_gui import main_camera_thread_gui

from main_record_thread import main_record_thread

# functions


def exit(event):
    print("My soical app is closing ")
    engine = pyttsx3.init()
    engine.say("Thank you for using My Social eye")
    engine.say("the program is closing")
    engine.runAndWait()
    # close the root and end session
    root.destroy()


def welcomeMain():
    engine = pyttsx3.init()
    engine.say("Welcome you are in home page ")
    engine.say("press control button to know the  guidelines")
    engine.runAndWait()


def welcomeMeeting():
    engine = pyttsx3.init()
    engine.say("you have activated meeting analysis mode ")
    engine.say("press the Q button to switch it off")
    engine.runAndWait()


def welcomeCamera():
    engine = pyttsx3.init()
    engine.say("you are in Camera Mode page ")
    engine.say("press the S button to switch it off")
    engine.runAndWait()


def ByeCamera():
    engine = pyttsx3.init()
    engine.say("you are Leaving the Camera Mode page ")
    engine.say("you are returing to homepage")
    engine.say("press control button to know the  guidelines")
    engine.runAndWait()
    ins.config(text="Press (X --> activte meeting mode / C --> camera mode / Ctrl --> guidelines  / Z --> know what is my social eye / Escape --> close)")


def ByeMeeting(event):
    engine = pyttsx3.init()
    engine.say("you are Leaving the meeting analysis mode ")
    engine.say("you are returing to homepage")
    engine.say("press control button to know the  guidelines")
    engine.runAndWait()
    ins.config(text="Press (X --> activte meeting mode / C --> camera mode / Ctrl --> guidelines  / Z --> know what is my social eye / Escape --> close)")
    # TODO End the facial analysis here


def instructionss(event):
    engine = pyttsx3.init()
    engine.say("Hello welcome to my social eye app")
    engine.say("to go to meeting analysis mode  press the x button ")
    engine.say("to go to Camera Mode press the c  button ")
    engine.say("press control button to know the  guidelines")
    engine.say("to know what is my social eye app  press the z  button ")
    engine.say("to exit the program  press Escape  button ")
    engine.runAndWait()


def play(event):
    engine = pyttsx3.init()
    engine.say("my social eye app is an app that helps you to know how other participants are feelng in the online meeting by analyzing their facial experisions ")
    engine.runAndWait()


# go to meeting option
# window will be minimized
# output voice that the meeting analysis mode is on
# to end it at any time open the application and press ctrl + s to stop and return to main page
def activateMeetingAnalytics(event):
    print("activated Meeting Analytics ")
    welcomeMeeting()
    # change the text to tell that
    ins.config(text="Activated Meeting Analysis mode to stop press q button")
    # # minimize the root
    # root.wm_state('iconic')
    # root.iconify()
    main_record_thread(ins,root)


# open camera option
# add a frame that has a vide player
# output voice that the camera mode testing is on
# to end it at any time  press ctrl + s to stop and return to main page
def activateTestingCamera(event):
    print("activated Testing Camera")
    welcomeCamera()
    # open camera in the app
    # VideoCam(root,label,logo)
    main_camera_thread_gui(isCamera=True, root=root,
                           label=label, logo=logo, ins=ins)
    ins.config(text="Activated Camera mode to stop press s button")
    # ending the video cam
    ByeCamera()


# main page frame

# buliding application
HEIGHT = 700
WIDTH = 800
engine = pyttsx3.init()
root = tk.Tk()
root.title(" My Social Eye Desktop App ")
root.iconbitmap("logogp.png")
root.configure(bg='white', height=HEIGHT, width=WIDTH)
root.rowconfigure(0, weight=1)
root.columnconfigure(0, weight=1)


# buliding the frame
frameMain = LabelFrame(root, text="Welcoming Frame",
                       padx=50, pady=50, bg="white")
frameMain.pack(padx=10, pady=10)

# logo
logo = Image.open("logogp.png")
logo = ImageTk.PhotoImage(logo)
label = Label(frameMain, image=logo)
label.pack()

# written instructions
ins = Label(frameMain, text="Press (X --> activte meeting mode / C --> camera mode / Ctrl --> guidelines  / Z --> know what is my social eye / Escape --> close) ")
ins.pack()
# voice instructions
welcomeMain()


# keyshortcuts
root.bind("<Control_L>", instructionss)  # to get help guidelines
root.bind("z", play)  # to know what is my sociap eye
# to go to  activate the meeting analytics
root.bind("<x>", activateMeetingAnalytics)
root.bind("<c>", activateTestingCamera)  # to  open camera
root.bind("<Escape>", exit)  # to exit program exit button from escape
root.bind("<q>", ByeMeeting)

# run the program
root.mainloop()

# End of the frame


# from tkinter import ttk
# LARGEFONT =("Verdana", 35)
# HEIGHT = 700
# WIDTH = 800
# class tkinterApp(tk.Tk):

#     # __init__ function for class tkinterApp
#     def __init__(self, *args, **kwargs):

#         # __init__ function for class Tk
#         tk.Tk.__init__(self, *args, **kwargs)

#         # creating a container
#         container = tk.Frame(self)
#         container.pack(side = "top", fill = "both", expand = True)

#         container.grid_rowconfigure(0, weight = 1)
#         container.grid_columnconfigure(0, weight = 1)

#         # initializing frames to an empty array
#         self.frames = {}

#         # iterating through a tuple consisting
#         # of the different page layouts
#         for F in (StartPage, Page1, Page2):

#             frame = F(container, self)

#             # initializing frame of that object from
#             # startpage, page1, page2 respectively with
#             # for loop
#             self.frames[F] = frame

#             frame.grid(row = 0, column = 0, sticky ="nsew")

#         self.show_frame(StartPage)

#     # to display the current frame passed as
#     # parameter
#     def show_frame(self, cont):
#         frame = self.frames[cont]
#         frame.tkraise()

# first window frame startpage
# class StartPage(tk.Frame):
#     def __init__(self, parent, controller):
#         tk.Frame.__init__(self, parent)

#         # label of frame Layout 2
#         label = ttk.Label(self, text ="Startpage", font = LARGEFONT)

#         # putting the grid in its place by using
#         # grid
#         label.grid(row = 0, column = 4, padx = 10, pady = 10)

#         button1 = ttk.Button(self, text ="Page 1",
#         command = lambda : controller.show_frame(Page1))

#         # putting the button in its place by
#         # using grid
#         button1.grid(row = 1, column = 1, padx = 10, pady = 10)

#         ## button to show frame 2 with text layout2
#         button2 = ttk.Button(self, text ="Page 2",
#         command = lambda : controller.show_frame(Page2))

#         # putting the button in its place by
#         # using grid
#         button2.grid(row = 2, column = 1, padx = 10, pady = 10)

# # second window frame page1
# class Page1(tk.Frame):

#     def __init__(self, parent, controller):

#         tk.Frame.__init__(self, parent)
#         label = ttk.Label(self, text ="Page 1", font = LARGEFONT)
#         label.grid(row = 0, column = 4, padx = 10, pady = 10)

#         # button to show frame 2 with text
#         # layout2
#         button1 = ttk.Button(self, text ="StartPage",
#                             command = lambda : controller.show_frame(StartPage))

#         # putting the button in its place
#         # by using grid
#         button1.grid(row = 1, column = 1, padx = 10, pady = 10)

#         # button to show frame 2 with text
#         # layout2
#         button2 = ttk.Button(self, text ="Page 2",
#                             command = lambda : controller.show_frame(Page2))

#         # putting the button in its place by
#         # using grid
#         button2.grid(row = 2, column = 1, padx = 10, pady = 10)


# # third window frame page2
# class Page2(tk.Frame):
#     def __init__(self, parent, controller):
#         tk.Frame.__init__(self, parent)
#         label = ttk.Label(self, text ="Page 2", font = LARGEFONT)
#         label.grid(row = 0, column = 4, padx = 10, pady = 10)

#         # button to show frame 2 with text
#         # layout2
#         button1 = ttk.Button(self, text ="Page 1",
#                             command = lambda : controller.show_frame(Page1))

#         # putting the button in its place by
#         # using grid
#         button1.grid(row = 1, column = 1, padx = 10, pady = 10)

#         # button to show frame 3 with text
#         # layout3
#         button2 = ttk.Button(self, text ="Startpage",
#                             command = lambda : controller.show_frame(StartPage))

#         # putting the button in its place by
#         # using grid
#         button2.grid(row = 2, column = 1, padx = 10, pady = 10)

# # Driver Code
# app = tkinterApp()
# app.mainloop()

#########################################################################################################################
# from argparse import Action
# from logging import root
# from tkinter import *
# import tkinter as tk
# import PIL
# from PIL import ImageTk, Image
# from gtts import gTTS
# import pyttsx3
# import os
# ####functions
# def exit(event):
#     print("My soical app is closing ")


# def welcome():
#     engine = pyttsx3.init()
#     engine.say("press ctrl to continue")
#     engine.runAndWait()


# def instructionss(event):
#     engine = pyttsx3.init()
#     engine.say("Hello welcome to my social eye app")
#     engine.say(" to go to main page click  x button  ")
#     engine.say("to know wht you should use my social eye press the z  button ")
#     engine.say("to close the program  press EScape  button ")
#     engine.runAndWait()

# def play(event):
#     engine = pyttsx3.init()
#     engine.say("my social eye app is an app that helps you to know how other participants are feelng in the online meeting by analyzing their facial experisions ")
#     engine.runAndWait()

# ## open to new window  to the home page that has 2 buttons and 2 shortcuts
# def MainPage(event):
#     print("main page created")

# ## go to meeting option
# #window will be minimized
# #output voice that the meeting analysis mode is on
# # to end it at any time open the application and press ctrl + s to stop and return to main page
# def goToMeeting(event):
#     print("go to meetig button is clicked ")


# ## open camera option
# ## open a new window that opens the camera in the window
# #output voice that the camera mode testing is on
# ## to end it at any time  press ctrl + s to stop and return to main page
# def openCam(event):
#     print("open the camera in the application")


# ###buliding application
# HEIGHT = 700
# WIDTH = 800
# engine = pyttsx3.init()
# root=tk.Tk()
# root.title(" My Social Eye Desktop App ")
# root.iconbitmap("logogp.png")
# root.configure(bg='white')


# canvas = tk.Canvas(root ,height = HEIGHT, width=WIDTH,background="white")
# canvas.grid(columnspan=3, rowspan=3)


# ###logo
# logo = Image.open("logogp.png")
# logo=ImageTk.PhotoImage(logo)
# logo_label= tk.Label(image=logo)
# logo_label.image =logo
# logo_label.grid(column=1,row=0)


# ### written instructions
# ins = tk.Label(root ,text= "Please press crl button on the keyboard to continue // press Escape  button to close ")
# ins.grid(columnspan=3,column=0,row=1)


# ## voice instructions
# welcome()

# ###keyshortcuts
# root.bind("e",exit) # to exit program
# root.bind("<Control_L>",instructionss) # to READ program GUIDLINES
# root.bind("z",play) # to know what is my sociap eye
# root.bind("<x>",MainPage) # to go to main page

# ## run the program
# root.mainloop()
