# Gui by rootatkali
#import and using the thinker library to use the Gui
# to instal the library:
#   sudo dnf install python3-tkinter   --- on fedora

import tkinter as tk
from inference_classifier import inferenceClasFunct


window = tk.Tk()
window.title("semiamexslp")
inferenceClasFunct()

wndow.mainloop()

