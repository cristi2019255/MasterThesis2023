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
import threading
import time
#from playsound import playsound
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


TIMER_WARNING_INTERVAL = 10
TIMER_DURATION = 5


def run_timer(update_callback = lambda msg, time_up: print(msg), 
              stop_event: threading.Event = threading.Event(), 
              timer_duration: int = TIMER_DURATION, 
              timer_warning_interval: int = TIMER_WARNING_INTERVAL, 
              end_play_sound_path: str | None = None,
              indefinite_timer: bool = False
              ):
    """Runs a timer and plays a sound 5 times when time elapsed
    
    Args:
        update_callback (function: (message: str, time_up: bool) -> None, optional): the function to be executed once per second. Defaults to lambda msg: print(msg)
        stop_event (threading.Event, optional): stop timer event. Defaults to threading.Event()
        timer_duration (int, optional): The duration of the timer in seconds. Defaults to TIMER_DURATION
        timer_warning_interval (int, optional): The interval in seconds in which a sound is played after the expiration of the timer_duration
        end_play_sound_path (str, optional): The path to the sound file to play when time elapsed. Defaults to None
        indefinite_timer (bool, optional): The type of timer, if is set to False the time will stop after the timer_duration, else it will run until stop_event. Defaults to False
    WARNING:
        This function should run in a separate thread in order to not block the main thread
        
    Example:
        # Create a time stop event
        stop_timer_event = threading.Event()
        # Create a separate thread for the timer
        timer_thread = threading.Thread(target=run_timer, args=(update_callback=..., stop_event=stop_timer_event, timer_duration=60, timer_warning_interval=10, end_play_sound_path=..., indefinite_timer = True))
        # Start the timer thread
        timer_thread.start()

        ...
        
        # Stop the timer thread
        stop_timer_event.set()  

    """

    def play_sound(filename: str, amount=1):
        return
        # NB: playsound lib doesn't work so this function is doing nothing
        #for _ in range(amount):
        #    continue 
        #    playsound(filename)

    sound_thread = None
    start_time = time.time()

    while not stop_event.is_set():
        elapsed_time = time.time() - start_time
        remaining_time = int(timer_duration - elapsed_time)
        if (end_play_sound_path is not None) and ((remaining_time == 0) or (remaining_time < 0 and remaining_time % timer_warning_interval == 0)):
            if sound_thread is None:
                sound_thread = threading.Thread(target=play_sound, args=(end_play_sound_path, 5))
                sound_thread.start()
            elif not sound_thread.is_alive():
                sound_thread = threading.Thread(target=play_sound, args=(end_play_sound_path, 5))
                sound_thread.start()
        
        if not indefinite_timer and remaining_time <= 0:
            stop_event.set()

        minutes, seconds = divmod(abs(remaining_time), 60)
        time_up = remaining_time < 0
        sign = "-" if time_up else ""
        update_callback(f"Remaining Time: {sign}{minutes:02}:{seconds:02}", time_up)
        time.sleep(1)
        
    if sound_thread is not None:
        sound_thread.join()

    