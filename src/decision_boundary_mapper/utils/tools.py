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
import time
from .. import LoggerInterface

def track_time_wrapper(logger: LoggerInterface = None):
    """ Decorator that tracks the time it takes for a function to run.

    Args:
        logger (LoggerInterface): The logger to use. If None, prints to stdout.
    """
    def function_wrapper(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            result = func(*args, **kwargs)
            end = time.time()
            if logger is None:
                print(f"{func.__name__} took {end-start} seconds")
            else:
                logger.log(f"{func.__name__} took {end-start} seconds")
            return result
        return wrapper
    return function_wrapper
