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

from PIL import Image
import os
import numpy as np

def resize_images(dir:str, shape:int = 28):
    resized_name = f'resized_{shape}'
    resized_dir = dir.replace('resized', resized_name)
    if not os.path.exists(resized_dir):
        os.makedirs(resized_dir)
    
    for f in os.listdir(dir):
        new_filename = os.path.join(resized_dir, f)
        img = Image.open(os.path.join(dir, f))
        resized_img = img.resize((shape, shape))
        resized_img.save(new_filename) 
        
def convert_to_grayscale_images(dir:str):
    grayscale_dir = dir + "_grayscale"
    if not os.path.exists(grayscale_dir):
        os.makedirs(grayscale_dir)
    
    for f in os.listdir(dir):
        new_filename = os.path.join(grayscale_dir, f)
        grayscale_img = Image.open(os.path.join(dir, f)).convert('L')
        grayscale_img.save(new_filename)
    

resize_images('./data/parasites_focus_plane_divided/proto/resized', shape=28)
#convert_to_grayscale_images('./data/parasites_focus_plane_divided/proto/resized_64')