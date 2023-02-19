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
from tensorflow.keras.callbacks import Callback

class LoggerModel(Callback):
    def __init__(self, active:bool=True, name:str="Neural Network", info_color:str="magenta", show_init:bool=True, epochs:int=100):
        """ Initialize the logger

        Args:
            active (bool, optional): Defaults to True.
            name (str, optional): Defaults to "Logger".
        """
        self.active = active
        self.name = name
        self.info_color = info_color
        self.epochs = epochs
        if show_init:    
            sep = "=" * 30
            time = datetime.now().strftime("%H:%M:%S:%f")
            print(colored(f"[INFO] [{time}] [{self.name}] {sep}", self.info_color))
            print(colored(f"[INFO] [{time}] [{self.name}] {self.name} initialized", self.info_color))
            print(colored(f"[INFO] [{time}] [{self.name}] {sep}", self.info_color))
        

    def on_epoch_end(self, epoch, logs={}):
        if self.active:
            logs = ", ".join([f"{key}: {value:.4f}" for key, value in logs.items()])
            print(colored(f"[INFO] [{self.name}] [Epoch {epoch}/{self.epochs}] {logs}", self.info_color))
            