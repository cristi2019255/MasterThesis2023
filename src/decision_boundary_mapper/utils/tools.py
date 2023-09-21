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
import os
import time
from .. import LoggerInterface


def track_time_wrapper(logger: LoggerInterface | None = None):
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

def generate_class_name_mapper(file: str):
    mapper = {}
    with open(file, "r") as f:
        lines = f.readlines()
    for line in lines:
        class_num, class_name = int(line.split(" ")[0]) - 1, line.split(" ")[1].replace("\n", "")
        mapper[class_num] = class_name
    
    def class_name_mapper(x: int):
        return mapper[x]
    return class_name_mapper

def get_latest_created_file_from_folder(folder_path: str) -> str:
    """
    Get the latest created file in a folder
    
    Args:
        folder_path (str): The folder in which to search for the latest created file
    
    Returns:
        latest_file_path (str): The path to the latest created file in the folder
    
    Throws:
        Exception if no file in the folder
    """
    # Get a list of all files in the folder
    files = os.listdir(folder_path)

    # Filter out subdirectories (if any) and get the latest created file
    latest_file = None
    latest_timestamp = 0

    for file in files:
        file_path = os.path.join(folder_path, file)

        # Check if it's a file (not a directory)
        if os.path.isfile(file_path):
            file_timestamp = os.path.getctime(file_path)

            # Compare the file's creation timestamp to the latest found
            if file_timestamp > latest_timestamp:
                latest_timestamp = file_timestamp
                latest_file = file_path

    if latest_file is None:
        raise Exception("Could not find latest file")

    return latest_file
