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


import PySimpleGUI as sg
from ..utils import APP_FONT

def Collapsible(layout, key, title='', arrows=(sg.SYMBOL_UP, sg.SYMBOL_DOWN), collapsed=False, visible=True):
    """
    User Defined Element
    A "collapsable section" element. Like a container element that can be collapsed and brought back
    :param layout:Tuple[List[sg.Element]]: The layout for the section
    :param key:Any: Key used to make this section visible / invisible
    :param title:str: Title to show next to arrow
    :param arrows:Tuple[str, str]: The strings to use to show the section is (Open, Closed).
    :param collapsed:bool: If True, then the section begins in a collapsed state
    :return:sg.Column: Column including the arrows, title and the layout that is pinned
    """
    return sg.Column([
                      [sg.T((arrows[1] if collapsed else arrows[0]), enable_events=True, k=key+'/-BUTTON-', expand_x=True, font=APP_FONT),
                       sg.T(title, enable_events=True, key=key+'/-TITLE-', expand_x=True, font=APP_FONT)],
                      [sg.pin(
                            sg.Column(layout, key=key, visible=not collapsed, metadata=arrows, expand_x=True, expand_y=True), 
                          shrink=True, expand_x=True, expand_y=True)]], 
                     pad=(0,0), visible=visible, expand_x=True)
