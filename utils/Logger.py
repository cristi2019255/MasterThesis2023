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
from utils.LoggerInterface import LoggerInterface

class Logger(LoggerInterface):
    def __init__(self, active=True, name="Logger"):
        self.active = active
        self.name = name
        
        sep = "=" * 40
        time = datetime.now().strftime("%H:%M:%S:%f")
        print(colored(f"[{self.name}] [INFO] [{time}] {sep}", "magenta"))
        print(colored(f"[{self.name}] [INFO] [{time}] {self.name} initialized", "magenta"))
        print(colored(f"[{self.name}] [INFO] [{time}] {sep}", "magenta"))
        
        
    def log(self, message):
        """ Log a message to the console
            __param__ message: the message to log
        """
        if self.active:
            time = datetime.now().strftime("%H:%M:%S:%f")
            print(colored(f"[{self.name}] [INFO] [{time}] {message}", "magenta"))
    
    def warn(self, message):
        if self.active:
            time = datetime.now().strftime("%H:%M:%S:%f")
            print(colored(f"[{self.name}] [WARNING] [{time}] {message}", "yellow"))
    
    def error(self, message):
        if self.active:
            time = datetime.now().strftime("%H:%M:%S:%f")
            print(colored(f"[{self.name}] [ERROR] [{time}] {message}", "red"))
    
    def debug(self, message):
        if self.active:
            time = datetime.now().strftime("%H:%M:%S:%f")
            print(colored(f"[{self.name}] [DEBUG] [{time}] {message}","blue"))
            
    def success(self, message):
        if self.active:
            time = datetime.now().strftime("%H:%M:%S:%f")
            print(colored(f"[{self.name}] [SUCCESS] [{time}] {message}", "green"))