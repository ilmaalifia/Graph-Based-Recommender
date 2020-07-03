from tkinter import *
import tkinter.messagebox
import os
import subprocess as sub
import sys

def run(k, iterations, file):
    if k.isnumeric() and iterations.isnumeric():
        command = 'python ' + file + '.py ' + k + ' ' + iterations
        os.system(command)
    else:
        tkinter.messagebox.showinfo("Peringatan", "Nilai parameter harus numerik!")

file = sys.argv[1]

master = Tk()
master.title('Algoritma SCLUB-CD')
frame = Frame(master) 
frame.pack(fill = BOTH, expand = True) 

bottomframe = Frame(master)
bottomframe.pack(fill = BOTH, expand = True, side = BOTTOM)

bottomtopframe = Frame(bottomframe)
bottomtopframe.pack(fill = BOTH, expand = True, side = TOP)

bottomdownframe = Frame(bottomframe)
bottomdownframe.pack(fill = BOTH, expand = True, side = BOTTOM)

resultframe = Frame(bottomdownframe)
resultframe.pack(fill = BOTH, expand = True, side = BOTTOM)

k = StringVar()
iterations = StringVar()

Label(frame, text = '', font='Arial 12').pack()
Label(frame, text = '   Perbandingan Deteksi Komunitas   ', font='Arial 12 bold').pack()
Label(frame, text = '   Algoritma Louvain dan Algoritma GN-MC   ', font='Arial 12 bold').pack()
Label(frame, text = '', font='Arial 12').pack()

Label(frame, text = '         k         ', font='Arial 10 italic').pack(side = LEFT, expand = True, fill = BOTH)
Entry(frame, textvariable = k).pack(side = LEFT, expand = True, fill = BOTH)
Label(frame, text = '', font='Arial 12').pack(side = RIGHT)

Label(bottomtopframe, text = 'Jumlah Iterasi', font='Arial 10').pack(side = LEFT, expand = True, fill = BOTH)
Entry(bottomtopframe, textvariable = iterations).pack(side = LEFT, expand = True, fill = BOTH)
Label(bottomtopframe, text = '', font='Arial 12').pack(side = RIGHT)

def k_callback(*args):
    print ('k = ', k.get())
def iterations_callback(*args):
    print ('iterations = ',iterations.get())

k.trace("w", k_callback)
iterations.trace("w", iterations_callback)

Label(bottomdownframe, text = '', font='Arial 12').pack()
button = Button(bottomdownframe, bg = 'green', fg = 'white', text = 'JALANKAN', font='Arial 10 bold', width = 20, height = 2, activebackground = 'gray', activeforeground = 'white', command = lambda: run(k.get(), iterations.get(), file))
button.pack(expand = True)
Label(bottomdownframe, text = '', font='Arial 12').pack()

mainloop()