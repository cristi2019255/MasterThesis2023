<!--
 Copyright 2023 Cristian Grosu
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
-->

In order to deploy the python package follow the steps:

1) Run `python3 setup.py sdist`
2) Run `twine upload dist/*`
   2.1) For the username use `__token__`
   2.2) For the password use `pypi-` + the token you get from the pypi website
3) Provide your credentails from pypi
4) Download the package by running `pip install decision-boundary-mapper`
