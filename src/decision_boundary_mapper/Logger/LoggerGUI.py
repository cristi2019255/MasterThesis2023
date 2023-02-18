# Copyright 2023 Cristian Grosu
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

class LoggerGUI(LoggerInterface):
    """ Logs the messages to the GUI.
    """
    
    def __init__(self, name:str="Logger", active:bool=True, output=None, update_callback=lambda x: x, info_color:str="magenta", show_init:bool=True):
        """ Initialize the logger

        Args:
            name (str, optional): Defaults to "Logger".
            active (bool, optional): Defaults to True.
            output (any, optional): The GUI reference where the logger should display messages. Defaults to None.
            update_callback (python function, optional): The function that is called to update the GUI. Defaults to lambda x:x.
        """
        super().__init__()

        self.active = active 
        self.name = name
        self.info_color = info_color
    
        if output is not None:
            self.output = output
            self.update_callback = update_callback
        else:
            print("No output provided for LoggerGUI. LoggerGUI will not print anything.")
            self.active = False
        
        if show_init and self.active:    
            sep = "=" * 40
            time = datetime.now().strftime("%H:%M:%S:%f")
            
            self.print(f"[{self.name}] [INFO] [{time}] {sep}", self.info_color)
            self.print(f"[{self.name}] [INFO] [{time}] {self.name} initialized", self.info_color)
            self.print(f"[{self.name}] [INFO] [{time}] {sep}", self.info_color)


    def print(self, message:str, color:str='magenta'):
        if self.active:
            self.output.print(message, text_color=color)
            self.update_callback()
            
    def log(self, message:str):
        time = datetime.now().strftime("%H:%M:%S:%f")
        self.print(f"[{self.name}] [INFO] [{time}] {message}", self.info_color)
    
    def warn(self, message:str):
        time = datetime.now().strftime("%H:%M:%S:%f")
        self.print(f"[{self.name}] [WARNING] [{time}] {message}", "yellow")
    
    
    def error(self, message:str):
        time = datetime.now().strftime("%H:%M:%S:%f")
        self.print(f"[{self.name}] [ERROR] [{time}] {message}", "red")        
    
    def debug(self, message:str):
        time = datetime.now().strftime("%H:%M:%S:%f")
        self.print(f"[{self.name}] [DEBUG] [{time}] {message}","blue")
    
        
    def success(self, message:str):
        time = datetime.now().strftime("%H:%M:%S:%f")
        self.print(f"[{self.name}] [SUCCESS] [{time}] {message}", "green")