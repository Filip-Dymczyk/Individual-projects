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
        self.__window = tk.Tk()
        self.__window.title("Bank account")

        # Entry fields needed to create user:
        self.__entry_login = tk.Entry(self.__window, width=30)
        self.__entry_password = tk.Entry(self.__window, width=30)

        # Creating associated user:
        self.__user: User = User()

    def __configure_window(self) -> None:
        # Setting pad on y axis:
        pady = 5

        # First label over login field:
        label1 = tk.Label(self.__window, text="Login")
        label1.pack(pady=pady)

        # Packing login entry:
        self.__entry_login.pack(pady=pady)

        # Second label over password field:
        label2 = tk.Label(self.__window, text="Password")
        label2.pack(pady=pady)

        # Packing password entry:
        self.__entry_password.pack(pady=pady)

        # Button for confirming login and password - calls a __submit_login_password on press:
        confirm_button = tk.Button(self.__window, text="Confirm", command=self.__submit_login_password)
        confirm_button.pack(pady=pady)

        # Button for closing the window:
        close_button = tk.Button(self.__window, text="Close", command=self.__terminate)
        close_button.pack(pady=pady)

    def __submit_login_password(self) -> None:
        self.__user.set_login(self.__entry_login.get())
        self.__user.set_password(self.__entry_password.get())

    def get_user_login(self) -> str:
        return self.__user.get_login()

    def activate(self) -> None:
        self.__configure_window()
        self.__window.mainloop()

    def __terminate(self) -> None:
        self.__window.destroy()