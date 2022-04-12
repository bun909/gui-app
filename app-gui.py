from Detector import main_app
from create_classifier import train_classifer
from create_dataset import start_capture
import tkinter as tk
from tkinter import font as tkfont
from tkinter import messagebox,PhotoImage
from PIL import ImageTk, Image
from gender_prediction import emotion, ageAndgender
names = set()


class MainUI(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        global names
        with open("nameslist.txt", "r") as f:
            x = f.read()
            z = x.rstrip().split(" ")
            for i in z:
                names.add(i)
        self.title_font = tkfont.Font(family='Helvetica', size=16, weight="bold")
        self.title("Nhan Dien Khuon Mat")
        self.resizable(False, False)
        self.geometry("500x250")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.active_name = None
        container = tk.Frame(self)
        container.grid(sticky="nsew")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames = {}
        for F in (StartPage, PageOne, PageTwo, PageThree, PageFour):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame("StartPage")

    def show_frame(self, page_name):
            frame = self.frames[page_name]
            frame.tkraise()

    def on_closing(self):

        if messagebox.askokcancel("Thoat", "Ban Chac Chu?"):
            global names
            f =  open("nameslist.txt", "a+")
            for i in names:
                    f.write(i+" ")
            self.destroy()


class StartPage(tk.Frame):

        def __init__(self, parent, controller):
            tk.Frame.__init__(self, parent)
            self.controller = controller
            load = Image.open("homepagepic.png")
            load = load.resize((250, 250), Image.ANTIALIAS)
            render = PhotoImage(file='homepagepic.png')
            img = tk.Label(self, image=render)
            img.image = render
            img.grid(row=0, column=1, rowspan=4, sticky="nsew")
            label = tk.Label(self, text="        Trang Chu        ", font=self.controller.title_font,fg="#263942")
            label.grid(row=0, sticky="ew")
            button1 = tk.Button(self, text="   Them nguoi dung  ", fg="#ffffff", bg="#263942",command=lambda: self.controller.show_frame("PageOne"))
            button2 = tk.Button(self, text="   Kiem tra nguoi dung  ", fg="#ffffff", bg="#263942",command=lambda: self.controller.show_frame("PageTwo"))
            button3 = tk.Button(self, text="Thoat", fg="#263942", bg="#ffffff", command=self.on_closing)
            button1.grid(row=1, column=0, ipady=3, ipadx=7)
            button2.grid(row=2, column=0, ipady=3, ipadx=2)
            button3.grid(row=3, column=0, ipady=3, ipadx=32)


        def on_closing(self):
            if messagebox.askokcancel("Thoat", "Ban Chac Chu?"):
                global names
                with open("nameslist.txt", "w") as f:
                    for i in names:
                        f.write(i + " ")
                self.controller.destroy()


class PageOne(tk.Frame):
    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        tk.Label(self, text="Nhap Ten", fg="#263942", font='Helvetica 12 bold').grid(row=0, column=0, pady=10, padx=5)
        self.user_name = tk.Entry(self, borderwidth=3, bg="lightgrey", font='Helvetica 11')
        self.user_name.grid(row=0, column=1, pady=10, padx=10)
        self.buttoncanc = tk.Button(self, text="Huy", bg="#ffffff", fg="#263942", command=lambda: controller.show_frame("StartPage"))
        self.buttonext = tk.Button(self, text="Tiep Theo", fg="#ffffff", bg="#263942", command=self.start_training)
        self.buttoncanc.grid(row=1, column=0, pady=10, ipadx=5, ipady=4)
        self.buttonext.grid(row=1, column=1, pady=10, ipadx=5, ipady=4)
    def start_training(self):
        global names
        if self.user_name.get() == "None":
            messagebox.showerror("Loi", "Ten Khong Duoc Trong!")
            return
        elif self.user_name.get() in names:
            messagebox.showerror("Loi", "Ten Da Ton Tai")
            return
        elif len(self.user_name.get()) == 0:
            messagebox.showerror("Loi", "Ten Khong Duoc Trong!")
            return
        name = self.user_name.get()
        names.add(name)
        self.controller.active_name = name
        self.controller.frames["PageTwo"].refresh_names()
        self.controller.show_frame("PageThree")


class PageTwo(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        global names
        self.controller = controller
        tk.Label(self, text="Chon Nguoi Dung", fg="#263942", font='Helvetica 12 bold').grid(row=0, column=0, padx=10, pady=10)
        self.buttoncanc = tk.Button(self, text="Huy", command=lambda: controller.show_frame("StartPage"), bg="#ffffff", fg="#263942")
        self.menuvar = tk.StringVar(self)
        self.dropdown = tk.OptionMenu(self, self.menuvar, *names)
        self.dropdown.config(bg="lightgrey")
        self.dropdown["menu"].config(bg="lightgrey")
        self.buttonext = tk.Button(self, text="Tiep Theo", command=self.nextfoo, fg="#ffffff", bg="#263942")
        self.dropdown.grid(row=0, column=1, ipadx=8, padx=10, pady=10)
        self.buttoncanc.grid(row=1, ipadx=5, ipady=4, column=0, pady=10)
        self.buttonext.grid(row=1, ipadx=5, ipady=4, column=1, pady=10)

    def nextfoo(self):
        if self.menuvar.get() == "None":
            messagebox.showerror("ERROR", "Name cannot be 'None'")
            return
        self.controller.active_name = self.menuvar.get()
        self.controller.show_frame("PageFour")

    def refresh_names(self):
        global names
        self.menuvar.set('')
        self.dropdown['menu'].delete(0, 'end')
        for name in names:
            self.dropdown['menu'].add_command(label=name, command=tk._setit(self.menuvar, name))

class PageThree(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        self.numimglabel = tk.Label(self, text="Anh da chup = 0 ", font='Helvetica 12 bold', fg="#263942")
        self.numimglabel.grid(row=0, column=0, columnspan=2, sticky="ew", pady=10)
        self.capturebutton = tk.Button(self, text="Chup du lieu anh", fg="#ffffff", bg="#263942", command=self.capimg)
        self.trainbutton = tk.Button(self, text="Dao Tao Du Lieu", fg="#ffffff", bg="#263942",command=self.trainmodel)
        button0 = tk.Button(self, text="Tro Ve Trang Chu", command=lambda: self.controller.show_frame("StartPage"), bg="#ffffff", fg="#263942")
        button0.grid(row=1,column=3, sticky="ew", ipadx=5, ipady=4, padx=10, pady=10)
        self.capturebutton.grid(row=1, column=0, ipadx=5, ipady=4, padx=10, pady=20)
        self.trainbutton.grid(row=1, column=1, ipadx=5, ipady=4, padx=10, pady=20)

    def capimg(self):
        self.numimglabel.config(text=str("Anh da chup = 0 "))
        messagebox.showinfo("Thong Bao", "He thong se chup 300 buc anh cua khuon mat ban")
        x = start_capture(self.controller.active_name)
        self.controller.num_of_images = x
        self.numimglabel.config(text=str("Anh da chup = "+str(x)))

    def trainmodel(self):
        if self.controller.num_of_images < 300:
            messagebox.showerror("Loi", "Khong Du Lieu, Vui Long Chup Toi Thieu 300 Buc Anh!")
            return
        train_classifer(self.controller.active_name)
        messagebox.showinfo("Thanh Cong", "He thong da duoc dao tao thanh cong!")
        self.controller.show_frame("PageFour")


class PageFour(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller

        label = tk.Label(self, text="       Tinh Nang      ", font='Helvetica 16 bold')
        label.grid(row=0,column=0, sticky="ew")
        button1 = tk.Button(self, text="Nhan Dien Nguoi Dung", command=self.openwebcam, fg="#ffffff", bg="#263942")
        button2 = tk.Button(self, text="Nhan Dang Cam Xuc", command=self.emot, fg="#ffffff", bg="#263942")
        button3 = tk.Button(self, text="Du Doan Gioi Tinh Va Tuoi", command=self.gender_age_pred, fg="#ffffff", bg="#263942")
        button4 = tk.Button(self, text="Tro Ve Trang Chu", command=lambda: self.controller.show_frame("StartPage"), bg="#ffffff", fg="#263942")
        button1.grid(row=1,column=0, sticky="ew", ipadx=5, ipady=4, padx=10, pady=10)
        button2.grid(row=1,column=1, sticky="ew", ipadx=5, ipady=4, padx=10, pady=10)
        button3.grid(row=2,column=0, sticky="ew", ipadx=5, ipady=4, padx=10, pady=10)
        button4.grid(row=1,column=1, sticky="ew", ipadx=5, ipady=4, padx=10, pady=10)

    def openwebcam(self):
        main_app(self.controller.active_name)
    def gender_age_pred(self):
       ageAndgender()
    def emot(self):
        emotion()



app = MainUI()
app.iconphoto(False, tk.PhotoImage(file='icon.ico'))
app.mainloop()

