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
        # Creating Tk-window with title:
        self.__window = tk.Tk()
        self.__window.title("Bank account")

        # Creating associated user:
        self.__user: User = User()

    def __configure_window_log_in(self) -> None:

        # Helper function triggering on confirm_button press - saves user data entered if they match system conditions:
        def submit_login_password() -> None:
            self.__user.set_login(entry_login.get())
            self.__user.set_password(entry_password.get())

        # Helper functions responsible for handling events generated by hovering buttons:
        def on_hover_enter_confirm(event: tk.Event):
            confirm_button.config(bg="green")

        def on_hover_enter_close(event: tk.Event):
            close_button.config(bg="red")

        def on_hover_leave_confirm(event: tk.Event):
            confirm_button.config(bg="white")

        def on_hover_leave_close(event: tk.Event):
            close_button.config(bg="white")

        # Setting pad on y axis:
        pady = 5

        # Labels:
        label1 = tk.Label(self.__window, text="Login")
        label2 = tk.Label(self.__window, text="Password")

        # Entry fields to enter login and password:
        entry_login = tk.Entry(self.__window, width=30)
        entry_password = tk.Entry(self.__window, show="*", width=30)

        # Buttons for: data confirmation and closing the window:
        confirm_button = tk.Button(self.__window, text="Confirm", command=submit_login_password)
        close_button = tk.Button(self.__window, text="Close", command=self.__terminate)

        # Introducing new events on entering and leaving confirm and close button area:
        confirm_button.bind("<Enter>", on_hover_enter_confirm)
        close_button.bind("<Enter>", on_hover_enter_close)
        confirm_button.bind("<Leave>", on_hover_leave_confirm)
        close_button.bind("<Leave>", on_hover_leave_close)

        # Packing created widgets into window:
        label1.pack(pady=pady)
        entry_login.pack(pady=pady)
        label2.pack(pady=pady)
        entry_password.pack(pady=pady)
        confirm_button.pack(pady=pady)
        close_button.pack(pady=pady)

    def get_user_login(self) -> str:
        return self.__user.get_login()

    def activate(self) -> None:
        self.__configure_window_log_in()
        self.__window.mainloop()

    def __terminate(self) -> None:
        self.__window.destroy()