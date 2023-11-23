from typing import Optional


# Class representing a database, probably file reader and writer at first; in later versions maybe a regular database:
class DataBase:
    def __init__(self) -> None:

        # Keeping track of information that may be used to speed-up later processes of the application:
        self.__account_balance: Optional[float] = None
        self.__current_login: Optional[str] = None
        self.__current_password: Optional[str] = None

        # Getting appropriate file path:
        file_path = __file__
        file_path = file_path.split("\\")
        file_path = file_path[:-1]
        file_path = "\\".join(file_path)
        file_path += "\\data.txt"
        self.__file_path = file_path

    # Checking if login exists in the file - we don't allow same logins:
    def check_login_in_file(self, login: str) -> bool:
        with open(self.__file_path, 'r') as file:
            return login in file.read()

    # Checking (during login) if the login entered is present and if it matches the password in the same
    # line of the file.
    # True means that login data are viable, False says otherwise.
    def check_login_password(self, login: str, password) -> bool:

        if self.check_login_in_file(login):

            # Opening file to read lines in order to search for user:
            with open(self.__file_path, 'r') as file:
                for i, line in enumerate(file.readlines()):

                    # if given login is in the line:
                    if login in line:

                        # We split it so that there are no spaces - we get an array:
                        line_split = line.split()

                        # Second element is password:
                        password_from_file = line_split[1]

                        # Third element is balance which we save:
                        self.__account_balance = float(line_split[2])

                        return password_from_file == password
        return False

    # Writing data to file after registration is finished. Remembers client's account balance.
    def write_data_to_file(self, login: str, password: str) -> None:
        with open(self.__file_path, "a+") as file:
            # Writing appropriately formatted line to file:
            line = f"{login} {password} {0}\n"
            file.write(line)

    # Updating balance after terminating account management window:
    def update_balance(self, balance: float) -> None:
        # Removing the old line from file:
        with open(self.__file_path, "r+") as file:
            new_f = file.readlines()
            file.seek(0)
            for line in new_f:
                if self.__current_login not in line:
                    file.write(line)
            file.truncate()
        # Writing updated line to file:
        with open(self.__file_path, "a") as file:
            new_line = f"{self.__current_login} {self.__current_password} {balance}\n"
            file.write(new_line)

    # Setting login and password of user that is currently logged in:
    def set_curr_login(self, login: str) -> None:
        self.__current_login = login

    def set_curr_password(self, password: str) -> None:
        self.__current_password = password

    # Returning balance related to logged user:
    def get_account_balance(self) -> float:
        return self.__account_balance


# Class representing account details and allowing for operations:
class Account:
    def __init__(self, balance: float = 0) -> None:
        self.__balance = balance

    def get_balance(self) -> float:
        return self.__balance

    def set_balance(self, balance: float) -> None:
        self.__balance = balance

    def withdraw(self, amount: float) -> None:
        self.__balance -= abs(amount)

    def deposit(self, amount: float) -> None:
        self.__balance += abs(amount)
