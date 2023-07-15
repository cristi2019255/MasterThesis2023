# Copyright 2022 Cristian Grosu
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from datetime import datetime
from termcolor import colored

from .LoggerInterface import LoggerInterface


class Logger(LoggerInterface):
    """ A console logger
        Prints the messages to the default console
    """

    def __init__(self, active: bool = True, name: str = "Logger", info_color: str = "magenta", show_init: bool = True):
        """ Initialize the logger

        Args:
            active (bool, optional): Defaults to True.
            name (str, optional): Defaults to "Logger".
            info_color (str, optional): Defaults to "magenta". The color of the info message
            show_init (bool, optional): Defaults to True. Show the initialization message
        """
        self.active = active
        self.name = name
        self.info_color = info_color

        if show_init:
            sep = "=" * 30
            time = datetime.now().strftime("%H:%M:%S")
            print(colored(f"[INFO] [{time}] [{self.name}] {sep}", self.info_color))
            print(colored(f"[INFO] [{time}] [{self.name}] {self.name} initialized", self.info_color))
            print(colored(f"[INFO] [{time}] [{self.name}] {sep}", self.info_color))

    def log(self, message: str):
        """ Log a message to the console
            Args:
                message (str): the message to log
        """
        if self.active:
            time = datetime.now().strftime("%H:%M:%S")
            print(colored(f"[INFO] [{time}] [{self.name}] {message}", self.info_color))

    def warn(self, message: str):
        if self.active:
            time = datetime.now().strftime("%H:%M:%S")
            print(colored(f"[WARNING] [{time}] [{self.name}] {message}", "yellow"))

    def error(self, message: str):
        if self.active:
            time = datetime.now().strftime("%H:%M:%S")
            print(colored(f"[ERROR] [{time}] [{self.name}] {message}", "red"))

    def debug(self, message: str):
        if self.active:
            time = datetime.now().strftime("%H:%M:%S")
            print(colored(f"[DEBUG] [{time}] [{self.name}] {message}", "blue"))

    def success(self, message: str):
        if self.active:
            time = datetime.now().strftime("%H:%M:%S")
            print(colored(f"[SUCCESS] [{time}] [{self.name}] {message}", "green"))
