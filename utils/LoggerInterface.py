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

class LoggerInterface:
    def log(self, message):
        """ Log a message to the console
        __param__ message: the message to log
        """
        pass
        
    def warn(self, message):
        pass
    
    def error(self, message):
        pass
    
    def debug(self, message):
        pass
        
    def success(self, message):
        pass