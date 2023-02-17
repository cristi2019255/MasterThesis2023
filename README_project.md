# :mag: Decision Boundary Mapper  (MasterThesis2023)

This is the code repository for master thesis of Grosu Cristian for 2023 at Utrecht University

## [:books: API Documentation](https://decisionboundarymapper.000webhostapp.com/)

## :scroll: Usage guide

If you are planning to use this repo in your project then simply install the package in your repository by running `pip install decision-boundary-mapper`

If you want to use the functionalities directly from this repo the follow the next flow:

1. Run the following command to install the needed dependencies: `pip install -r requirements.txt`
2. Run the following command to start the application `python3 mainGUI.py`
3. Run the following script to get an example of how the Decision Boundary Mapper works `python3 main.py`

## :dart: Tasks to be done (This is for internal use only)

1. Think about a way of how to upload the data to the GUI faster

This is not really an issue

2. Projection errors are taking too long to be generated
   I do not even know if the way they are computed now is the correct way ...

3.TODO:
change back the generate_fast_dbm method so it uses fixed nd data, no interpolation used for spaceNd
update the functions descriptions in DBM folder
see if Autoencoder results make sense
