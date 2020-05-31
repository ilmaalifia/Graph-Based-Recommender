import subprocess
import tkinter as tk
from threading import Thread

master = tk.Tk()

def sudo(cmd, terminal):

    # sudo_password = 'your sudo code' + '\n'
    # sudos = ['sudo', '-S']

    terminal.delete('1.0', tk.END)


    p = subprocess.Popen(cmd,  stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, universal_newlines=True)
    # p.stdin.write(sudo_password)
    p.poll()

    while True:
        line = p.stdout.readline()
        terminal.insert(tk.END, line)
        terminal.see(tk.END)
        # top.updates()
        if not line and p.poll is not None: break

    while True:
        err = p.stderr.readline()
        terminal.insert(tk.END, err)
        terminal.see(tk.END)
        # top.updates()
        if not err and p.poll is not None: break
    terminal.insert(tk.END, '\n * END OF PROCESS *')

    # p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1, universal_newlines=True, shell = True)
    # p.poll()

    # while True:
    #     line = p.stdout.readline()
    #     terminal.insert(tk.END, line)
    #     terminal.see(tk.END)
    #     if not line and p.poll is not None: break

    # while True:
    #     err = p.stderr.readline()
    #     terminal.insert(tk.END, err)
    #     terminal.see(tk.END)
    #     if not err and p.poll is not None: break
    # terminal.insert(tk.END, '\n Finished download')

textfield = tk.Text(master, font = "Arial 15")
textfield.pack()

link = "https://www.youtube.com/watch?v=s8XIgR5OGJc"
a = "youtube-dl --extract-audio --audio-format mp3 '{0}'".format(link)

t = Thread(target = lambda: sudo('python lastfm_test.py 10 100', textfield))
t.start()

tk.mainloop()