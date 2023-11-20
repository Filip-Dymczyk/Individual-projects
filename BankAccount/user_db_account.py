from typing import Optional


# Class representing user data who is currently using the program: login, password and an account:
class User:
    def __init__(self) -> None:
        self.__login: Optional[str] = None
        self.__password: Optional[str] = None
        self.account: Account = Account()

    def set_login(self, login: str) -> None:
        self.__login = login

    def set_password(self, password: str) -> None:
        self.__password = password

    def get_login(self) -> str:
        return self.__login

    def get_password(self) -> str:
        return self.__password


# Class representing a database, probably file reader and writer at first; in later versions maybe a regular database:
class DataBase:
    def __init__(self) -> None:

        # Keeping track of information that may be used to speed-up later processes of the application like:
        # number of line in which we have data related to the user that is currently logged and his account balance.
        self.line_nr: Optional[int] = None
        self.__account_balance: Optional[float] = None

        # File opening:
        self.__file = open("C:\\Users\\User\\Desktop\\Python_projects\\python_projects\\BankAccount\\data.txt", "r+")

    # Destructor - closing the file on app termination:
    def __del__(self) -> None:
        self.__file.close()

    # Checking if login exists in the file - we don't allow same logins:
    def check_login_in_file(self, login: str) -> bool:
        return login in self.__file.read()

    # Checking (during login) if the login entered is present and if it matches the password in the same
    # line of the file.
    # True means that login data are viable, False says otherwise.
    def check_login_password(self, login: str, password) -> bool:
        if self.check_login_in_file(login):
            # We get a tab of all lines and index of every line:
            for i, line in enumerate(self.__file.readlines()):
                # if given login is in the line:
                if login in line:
                    # We split it so that there are no spaces - we get an array:
                    line_split = line.split()
                    # Second element is password:
                    password_from_file = line_split[1]

                    # Third element is balance which we save:
                    self.__account_balance = float(line_split[2])
                    # We also get the line nr in which we have useful information:
                    self.line_nr = i

                    return password_from_file == password
        return False

    # Writing data to file after registration is finished. Remembers client's account balance.
    def write_data_to_file(self, login: str, password: str) -> None:
        line = f"{login} {password} {0}\n"
        self.__file.write(line)

        # It's now the last line - newly added:
        self.line_nr = len(self.__file.readlines()) - 1

        # Balance = 0:
        self.__account_balance = 0

    # Getting balance related to particular User:
    def get_balance(self) -> float:
        return self.__account_balance

    # Setting balance:
    def set_balance(self, balance: float) -> None:
        self.__account_balance = balance

    # Updating balance:
    def update_balance(self, login: str, password: str) -> None:
        pass


# Class representing account details:
class Account:
    def __init__(self) -> None:
        pass

