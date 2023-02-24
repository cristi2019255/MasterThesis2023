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

from matplotlib import pyplot as plt


def onclick(event):
    global point
    
    if event.inaxes == None:
        return
    
    x, y = event.xdata, event.ydata    
        
    if point is None:
        point = ax.plot(x, y, 'rx')[0]
        fig.canvas.draw_idle()
        return
    
    
    point.remove()
    
    point = None
    
    point = ax.plot(x, y, 'rx')[0] 
    fig.canvas.draw_idle()
    

def onkey(event):
    global point
    DELTA = 0.5
    if point is None:
        return
    
    (x,y) = point.get_data()
    print(x,y)
    if event.key == 'enter':
        point = None
        return
    if event.key == 'left':
        x -= DELTA
    if event.key == 'right':
        x += DELTA
    if event.key == 'up':
        y += DELTA
    if event.key == 'down':
        y -= DELTA
    
    if (event.key.isdigit()):
        print(event.key)
        print('int')
    point.remove()    
    point = ax.plot(x,y,'rx')[0]    
    fig.canvas.draw_idle()
    
 
point = None
fig = plt.figure()
ax = fig.add_subplot(111)
import numpy as np 
xx = np.linspace(0, 10, 100)
yy = np.sin(xx)
ax.plot(xx, yy)
fig.canvas.mpl_connect('button_press_event', onclick)
fig.canvas.mpl_connect('key_press_event', onkey)
plt.show()
