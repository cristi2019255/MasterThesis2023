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
from utils.LoggerInterface import LoggerInterface

class LoggerGUI(LoggerInterface):
    def __init__(self, name="Logger", active=True, output=None, update_callback=lambda x: x):
        super().__init__()

        self.active = active 
        self.name = name
        
        if output is not None:
            self.output = output
            self.update_callback = update_callback
        else:
            print("No output provided for LoggerGUI. LoggerGUI will not print anything.")
            self.active = False
            
        sep = "=" * 40
        time = datetime.now().strftime("%H:%M:%S:%f")
    
        self.print(f"[{self.name}] [INFO] [{time}] {sep}", "magenta")
        self.print(f"[{self.name}] [INFO] [{time}] {self.name} initialized", "magenta")
        self.print(f"[{self.name}] [INFO] [{time}] {sep}", "magenta")


    def print(self, message, color='magenta'):
        if self.active:
            self.output.print(message, text_color=color)
            self.update_callback()
            
    def log(self, message):
        time = datetime.now().strftime("%H:%M:%S:%f")
        self.print(f"[{self.name}] [INFO] [{time}] {message}", "magenta")
    
    def warn(self, message):
        time = datetime.now().strftime("%H:%M:%S:%f")
        self.print(f"[{self.name}] [WARNING] [{time}] {message}", "yellow")
    
    
    def error(self, message):
        time = datetime.now().strftime("%H:%M:%S:%f")
        self.print(f"[{self.name}] [ERROR] [{time}] {message}", "red")        
    
    def debug(self, message):
        time = datetime.now().strftime("%H:%M:%S:%f")
        self.print(f"[{self.name}] [DEBUG] [{time}] {message}","blue")
    
        
    def success(self, message):
        time = datetime.now().strftime("%H:%M:%S:%f")
        self.print(f"[{self.name}] [SUCCESS] [{time}] {message}", "green")