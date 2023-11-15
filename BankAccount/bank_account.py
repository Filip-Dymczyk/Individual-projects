import tkinter as tk
from typing import Optional


class User:
    def __init__(self) -> None:
        self.__login: Optional[str] = None
        self.__password: Optional[str] = None

    def set_login(self, login: str) -> None:
        self.__login = login

    def set_password(self, password: str) -> None:
        self.__password = password

    def get_login(self) -> str:
        return self.__login

    def get_password(self) -> str:
        return self.__password


class Account:
    def __init__(self) -> None:
        pass


class Interface:
    def __init__(self) -> None:
        # Creating Tk-window:
        self.__window = None

        # Creating associated user:
        self.__user: User = User()

    # Helper checking password or login viability:
    def __check_viability(self, l_p: str) -> bool:
        return 4 < len(l_p) < 10 and l_p.isalnum()

    # Functions handling buttons hovering:
    def __on_hover(self, event: tk.Event) -> None:
        widget = event.widget
        # Registration button:
        if widget.button_id == 1:
            pass
        # Confirm button:
        elif widget.button_id == 2:
            widget.config(bg="green")
        # Close button
        elif widget.button_id == 3:
            widget.config(bg="red")

    def __on_hover_leave(self, event: tk.Event) -> None:
        widget = event.widget
        widget.configure(bg="white")

    # Log-in window configuration:
    def __configure_window_log_in(self) -> None:

        # Initializing log_in window:
        self.__window = tk.Tk()
        self.__window.title("Log in")

        # Submitting login and password via confirm - initializing User:
        def submit_login_password() -> None:
            login = entry_login.get()
            password = entry_password.get()
            if self.__check_viability(login) and self.__check_viability(password):
                self.__user.set_login(login)
                self.__user.set_password(password)

        # Help functions responsible for handling events generated by pressing Enter on entries:
        def on_enter_pressed_login(event: tk.Event):
            login = entry_login.get()
            if self.__check_viability(login):
                entry_login.config(bg="green")
            else:
                entry_login.config(bg="red")

        def on_enter_pressed_password(event: tk.Event):
            password = entry_password.get()
            if self.__check_viability(password):
                entry_password.config(bg="green")
            else:
                entry_password.config(bg="red")

        # Setting pad on x- and y-axis:
        padx = 5
        pady = 5

        # Labels:
        label1 = tk.Label(self.__window, text="Login")
        label2 = tk.Label(self.__window, text="Password")

        # Entry fields to enter login and password:
        entry_login = tk.Entry(self.__window, width=30)
        entry_password = tk.Entry(self.__window, show="*", width=30)

        # Binding entry fields to trigger checking data entered in the fields by Enter:
        entry_login.bind("<Return>", on_enter_pressed_login)
        entry_password.bind("<Return>", on_enter_pressed_password)

        # Buttons for: registration, data confirmation and closing the window:
        registration_button = tk.Button(self.__window, text="Don't have an account yet? Click here to register.",
                                        command=self.__open_registration_window)
        confirm_button = tk.Button(self.__window, text="Confirm", command=submit_login_password)
        close_button = tk.Button(self.__window, text="Close", command=self.__terminate)

        # Introducing new events on entering and leaving confirm and close button area:
        confirm_button.bind("<Enter>", self.__on_hover)
        close_button.bind("<Enter>", self.__on_hover)
        confirm_button.bind("<Leave>", self.__on_hover_leave)
        close_button.bind("<Leave>", self.__on_hover_leave)

        # Adding id's:
        label1.label_id = 1
        label1.label_id = 2
        entry_login.entry_id = 1
        entry_password.entry_id = 2

        # Packing created widgets into window:
        label1.pack(pady=pady)
        entry_login.pack(pady=pady)
        label2.pack(pady=pady)
        entry_password.pack(pady=pady)
        confirm_button.pack(pady=pady)
        close_button.pack(side="right", padx=padx, pady=pady)

    def __open_registration_window(self) -> None:

        # Initializing registration window:
        self.__window = tk.Tk()
        self.__window.title("Register:")

        # Setting pad on x- and y-axis:
        padx = 5
        pady = 5

        # Labels:
        label1 = tk.Label(self.__window, text="Login")
        label2 = tk.Label(self.__window, text="Password")
        label3 = tk.Label(self.__window, text="Enter your password again")
        label4 = tk.Label(self.__window, text="Enter your email")

        # Entry fields to enter login and password and email:
        entry_login = tk.Entry(self.__window, width=30)
        entry_password = tk.Entry(self.__window, show="*", width=30)
        entry_re_password = tk.Entry(self.__window, show="*", width=30)
        entry_email = tk.Entry(self.__window, width=30)

        # # Binding entry fields to trigger checking data entered in the fields by Enter:
        # entry_login.bind("<Return>", on_enter_pressed_login)
        # entry_password.bind("<Return>", on_enter_pressed_password)


    def get_user_login(self) -> str:
        return self.__user.get_login()

    def activate(self) -> None:
        self.__configure_window_log_in()
        self.__window.mainloop()

    def __terminate(self) -> None:
        self.__window.destroy()
