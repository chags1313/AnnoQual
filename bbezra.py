# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 13:41:29 2020

@author: ColeHagen
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 15:30:29 2020

@author: ColeHagen
"""
import tkinter as tk
from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
import PIL.Image, PIL.ImageTk
import cv2
import pandas as pd
import time
from tkinter import PhotoImage
import subprocess

# For plots we need
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from tkinter import ttk

# TODO: add colors/style to matplotlib
# TODO: Allow for user to populate behavior categories at will
# TODO: Add function on_key_press
# TODO: Add ask before quit


# def on_key_press(event):
#    print("you pressed {}".format(event.key))
#    key_press_handler(event, canvas, toolbar)



class videoGUI:

    def __init__(self, window, window_title):

        window.iconbitmap(r"C:\Users\colehagen\Desktop\videoclass\openface\hnet.com-image.ico")        
        # region window creation
        self.window = window
        self.window.title(window_title)
        #self.window.configure(bg='black')
        
        menubar = tk.Menu(self.window)
        self.window.config(menu=menubar)
        subMenu = Menu(menubar, tearoff=0)
        subMenu2 = Menu(menubar, tearoff=0)
        subMenu3 = Menu(menubar, tearoff=0)
        subMenu4 = Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=subMenu)
        subMenu.add_command(label="Select Video", command=self.open_file)
        subMenu.add_command(label="Save Data", command= self.save_data)
        menubar.add_cascade(label="Automate", menu=subMenu2)
        subMenu2.add_command(label="Extract Action Units", command = self.open_face)
        menubar.add_cascade(label="Train", menu=subMenu3)
        subMenu3.add_command(label = "Select Training Data File (csv)", command = self.training_data)
        subMenu3.add_command(label = "Train Machine Learning Model", command = self.train_model)
        subMenu3.add_command(label = "Assess Performance", command = self.feature_graphs)
        menubar.add_cascade(label= "Predict", menu=subMenu4)
        subMenu4.add_command(label = "Select Prediction File (csv)", command=self.csvfile)
        subMenu4.add_command(label = "Run Machine Learning Model", command=self.runmodel)
        menubar.add_cascade(label= "Help", command = self.help)
        
        #tabControl = ttk.Notebook(window)
        #tab1 = ttk.Frame(tabControl)
        #tabControl.add(tab1, text = "Pain Codes")
        
        #tab2 = ttk.Frame(tabControl)
        #tabControl.add(tab2, text = "Pain Prediction")
        #tabControl.pack(expand=1, fill = 'both')
        
        bottom_frame = ttk.Frame(self.window)
        bottom_frame.pack(side=BOTTOM, pady=1)
        
        middle_frame = ttk.Frame(self.window)
        middle_frame.pack(side=BOTTOM, pady=1)
        
        top_frame = ttk.Frame(self.window)
        top_frame.pack(side=BOTTOM, pady=1)





        # Create pop-up for controls
        
        self.top = Toplevel()
        self.top.protocol("WM_DELETE_WINDOW", self.disable_event)
        self.top.iconbitmap(r"C:\Users\colehagen\Desktop\videoclass\openface\hnet.com-image.ico")
        # endregion

        # Region init variables

        self.pause = False   # Parameter that controls pause button

        self.canvas = Canvas(top_frame)
        self.canvas.pack()

        self.data = pd.DataFrame({'frameID':[], 'behavior':[]})

        self.annotate_flag = "off"
        

        # endregion


        # MAIN WINDOW
        # frame slider
        self.frame_slider = ttk.Scale(middle_frame, from_=0,
                                          orient=HORIZONTAL, length=500, command=self.get_slider_values)
        self.frame_slider.grid(row=0, column=1)
        #self.frame_slider.configure(bg = 'black')


        # Define start frame at 0
        self.current_frame = 0
        self.slider_pos = 0


        
        # Pause Button
        self.btn_pause=ttk.Button(middle_frame, text = ">/ll", width = 20, command=self.play_pause_video)
        self.btn_pause.grid(row=0, column=0)


        # Create Var for annotation

        self.var=StringVar()
        self.var.set('N/A')

        # Region radiobuttons

        self.b_dict = {'NA': 99, 'No Pain': 0, 'Pain': 1}

        for key in self.b_dict:
            self.b_dict[key] = Radiobutton(bottom_frame, text=key, height = 3, width=29, command=self.print_var)
            self.b_dict[key].config(indicatoron=0, variable=self.var, value=key)
            self.b_dict[key].pack(side='left')

        # endregion

        # TOP WINDOW

        # Button that lets the user take annotate
        
        self.btn_stop_annotate = ttk.Button(self.top, text="Stop Annotate", width=100, command=self.stop_annotate)
        self.btn_stop_annotate.pack(anchor=CENTER, expand=True, side = BOTTOM)
        
        self.btn_annotate = ttk.Button(self.top, text="Annotate", width=100, command=self.annotate)
        self.btn_annotate.pack(anchor=CENTER, expand=True, side = BOTTOM)
        
        # Play Button
        self.btn_play=ttk.Button(self.top, text="Start Video", width=100, command=self.play_video)
        self.btn_play.pack(anchor=CENTER, expand=True, side = BOTTOM)


        self.fig = Figure(figsize=(5, 4), dpi=100)


        #self.t = np.array(self.data)
        self.ax = self.fig.add_subplot(111)
        # Mind the comma! it's to get only one output
        self.ethogram, = self.ax.plot(self.data["frameID"], self.data["behavior"], '|', markersize=20)



        self.top_canvas = FigureCanvasTkAgg(self.fig, master=self.top)  # A tk.DrawingArea.
        self.top_canvas.draw()
        self.top_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)

        self.toolbar = NavigationToolbar2Tk(self.top_canvas, self.top)
        self.toolbar.update()
        self.top_canvas.get_tk_widget().pack(side=TOP, fill=BOTH, expand=1)





        self.delay = 15   # ms

        self.window.mainloop()

    def open_file(self):

        self.pause = False

        self.filename = filedialog.askopenfilename(title="Select file")
        print("Trying to read...")
        print(self.filename)

        # Open the video file
        self.cap = cv2.VideoCapture(self.filename)

        # Get the properties of the video
        self.width = self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)

        # Configure slider
        self.frame_slider.configure(to=self.total_frames)

        # Create space in the data
        self.data["frameID"] = range(0, int(self.total_frames))
        self.data["behavior"] = "NA"


        self.canvas.config(width = self.width, height = self.height)

    def get_slider_values(self, event):
        self.slider_pos = self.frame_slider.get()

        

    def annotate(self):

        if str(self.btn_annotate["state"]) == "active":
            # First disable button
            self.btn_annotate.configure(state=DISABLED)

        # Set a flag to mark you are annotating
        self.annotate_flag = "on"

        # Set values on data
        self.data.loc[int(self.current_frame), 'behavior'] = self.current_behavior


        # print(self.data.iloc[self.current_frame-1:self.current_frame+2])
        print("Annotating..." + self.current_behavior)


    def stop_annotate(self):
        # Enable annotate button back
        self.btn_annotate.configure(state=NORMAL)
        self.annotate_flag = "off"


    def print_var(self, *args):
        print("Changing to " + self.var.get())


    def get_frame(self):   # get only one frame

        try:

            if self.cap.isOpened():
                ret, frame = self.cap.read(self.current_frame)
                #self.current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        except:
            messagebox.showerror(title='End of Video', message='Please ensure all data has been entered. Verify with graph.')

    # This is the main function that we use to 'loop/update'
    def play_video(self):

        # First disable play button to prevent multiple instances of play
        self.disable_start_button()

        # Check if the slider has been moved
        if abs(self.slider_pos - self.current_frame) > 1:
            # Update the current frame
            self.current_frame = self.slider_pos
            # set the camera for that frame >>> THIS IS THE IMPORTANT TELL THE CAMERA WHAT"S THE NEXT FRAME!
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame - 1)

        # Get a frame from the video source, and go to the next frame automatically
        ret, frame = self.get_frame()

        # Show frame
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = NW)


        # Get the behavior
        self.current_behavior = self.var.get()
        # Call the annotate function
        if self.annotate_flag == "on":
            self.annotate()
        

        # update plot every 10 frames
        # We do this for speed purposes (we don't really need to refresh that fast)
        if self.current_frame % 10 == 0:
            self.refreshFigure()

        # If not pause, call this function again (aka, keep playing)
        if not self.pause:
            # increase the frame number
            self.current_frame = self.current_frame + 1

            # stop
            if self.current_frame > self.total_frames:
                self.pause = True
                messagebox.showerror(title='End of video', message='End of video!')

            # update the slider
            self.frame_slider.set(self.current_frame)
            self.window.after(self.delay, self.play_video)


    # This function serves as a toggle for play/pause
    def play_pause_video(self):
        if self.pause == False:
            self.pause = True
            self.play_video()
        else:
            self.pause = False
            self.play_video()
            
    def open_face(self):
        subprocess.Popen(["C:/Users/colehagen/Desktop/videoclass/openface/FaceLandmarkVidMulti.exe","-f", self.filename, "-vis-aus", "-vis-track"])
        tk.messagebox.showinfo('OpenFace Action Unit Extraction', 'OpenFace is currently extracting data from your video. This may take 5-15 mins. When the action unit plot window closes, all data will be available in the processed folder. Please reach out to Cole 320-237-0020 if you have any questions.')

    # We need to disable start so there are not a lot of instances
    def disable_start_button(self):
        self.btn_play.configure(state=DISABLED)

    # Release the video source when the object is destroyed
    def __del__(self):

        self.top.quit()

        if self.cap.isOpened():
            self.cap.release()

    # This will prevent the user from closing the top window (which stalls the program)
    def disable_event(self):
        pass

    def refreshFigure(self):

        # Clear the axis
        self.ax.clear()

        # new range
        x_min = max(0, self.current_frame - 500)
        x_max = min(self.current_frame + 500, self.total_frames)

        # We want them for indexing so change them to integers
        x_min = int(x_min)
        x_max = int(x_max)

        # Subset the portions we care about
        x = self.data["frameID"][x_min:x_max]
        y = self.data["behavior"][x_min:x_max]
            
            


        # make the new plot
        self.ax.plot(x, y, '|', markersize=20, color = 'g')

        self.ax = self.top_canvas.figure.axes[0]
        self.ax.set_xlim(x_min, x_max)
        # ax.set_ylim(y.min(), y.max())
        self.top_canvas.draw()

    def save_data(self):

        filename = filedialog.asksaveasfilename(title="Type filename to save",
                                                filetypes = (("csv files","*.csv"),("all files","*.*")))
        self.data.to_csv(filename, index=False)
    
    def help(self):
        tk.messagebox.showinfo('Help', 'Add your video by clicking in the upper left corner on Import Video. Begin the video by pressing Play Video under the graph or by pressing the >ll symbol under the video surface. Once the video is playing, select Annotate when you are ready to begin classifying behaviors in the frames. When annotation is on, click on the behavior that aligns with the video frame being viewed. To change a code, simply move the time slider back and select a different code. When finished coding, select the Save Data option in the upper left corner under File. If you would like to save your figure, click on the Save logo under the graph. Please reach out to Cole 320-237-0020 if you have questions. Thank you!')


    def training_data(self):
        self.trainfile = filedialog.askopenfilename(title="Select file",
                                                filetypes = (("csv files","*.csv"),("all files","*.*")))
        
    def train_model(self):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        
        df = pd.read_csv(self.trainfile, usecols= [1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
        
        X = df.drop('class', axis=1)
        y = df['class']
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
        
        from sklearn.svm import SVC
        svclassifier = SVC(kernel='linear')
        svclassifier.fit(X_train, y_train)
        
        y_pred = svclassifier.predict(X_test)
        
        from sklearn.metrics import classification_report, confusion_matrix
        print(confusion_matrix(y_test,y_pred))
        print(classification_report(y_test,y_pred))
        
        pred_df = pd.DataFrame()
        
        pred_df['class'] = y_pred

        #storing my predicted classes into my data frame
        
        trainsave = filedialog.asksaveasfilename(title="Type filename to save",
                                                filetypes = (("csv files","*.csv"),("all files","*.*")))
        
        pred_df.to_csv(trainsave)
        
        import numpy as np
        import matplotlib.pyplot as plt

        from sklearn import datasets
        from sklearn.metrics import plot_confusion_matrix

        # Plot non-normalized confusion matrix
        titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]
        for title, normalize in titles_options:
            disp = plot_confusion_matrix(svclassifier, X_test, y_test,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
            disp.ax_.set_title(title)

        print(title)
        print(disp.confusion_matrix)
        plt.savefig("confusion_matrix.png")
        plt.show()
        
        import pandas as pd
        import numpy as np
        np.random.seed(0)
        import matplotlib.pyplot as plt
        
        from sklearn.model_selection import train_test_split
        from sklearn import preprocessing
        from sklearn.ensemble import RandomForestRegressor
        # The target variable is 'quality'.
        Y = df['class']
        X = df.drop('class', axis=1)
        # Split the data into train and test data:
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)
            # Build the model with the random forest regression algorithm:
        model = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=10)
        model.fit(X_train, Y_train)
        import shap
        shap_values = shap.TreeExplainer(model).shap_values(X_train)
        shap.summary_plot(shap_values, X_train, plot_type="bar")
        shap.summary_plot(shap_values, X_train)
        
        
    def feature_graphs(self):
        print("hi")


    def csvfile(self):
        self.csvfile = filedialog.askopenfilename(title="Select file",
                                                filetypes = (("csv files","*.csv"),("all files","*.*")))
        
    def runmodel(self):
        import pandas as pd
        import numpy as np
        import matplotlib.pyplot as plt
        
        tk.messagebox.showinfo('SVM Pain Machine Learning Classifier', 'The machine learning model is currently running on your data. Please be sure to specify your file type as csv or xlsx. Time to complete predictions will vary, but most files take under 1 minute. Please allow enough time for model to complete. Pay attention for not when finished. Reach out to Cole with any questions.')
        
        df2 = pd.read_csv(self.csvfile, usecols= [0, 680, 681, 682, 683, 684, 685, 686, 687, 688, 689, 690, 691, 692, 693, 694, 695, 696])
        df = pd.read_csv("C:/Users/colehagen/Desktop/videoclass/aumodel/colepain.csv", usecols= [1, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
        
        X_train = df.drop('class', axis=1)
        y_train = df['class']
        X_test = df2.drop('class', axis=1)
        y_test = df2['class']
        
        from sklearn.svm import SVC
        svclassifier = SVC(kernel='linear')
        svclassifier.fit(X_train, y_train)
        
        y_pred = svclassifier.predict(X_test)
        
        pred_df2 = pd.DataFrame()
        
        pred_df2['class'] = y_pred

        #storing my predicted sale prices into my data frame
        
        
        savefile = filedialog.asksaveasfilename(title="Type filename to save",
                                                filetypes = (("csv files","*.csv"),("all files","*.*")))

        #storing my predicted classes into my data frame
        
        pred_df2.to_csv(savefile)
        
        view_pred = pd.read_csv(savefile)
        
        total_pain = view_pred['class'].sum()
        print(total_pain, "total frames were classified as pain after analysis by the machine learning model")
        
        
##### End Class #####


print("Starting app...")

time.sleep(1)


# Create a window and pass it to videoGUI Class
videoGUI(Tk(), "BBEzra Pain Coder")