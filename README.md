# IV Plotter for OSU Source Data

## Requirements
 - ROOT (PyROOT)
 - python packages -> ```pip install -r requirements.txt``` (or install manually from list in requirements.txt)

## Running
 - data has to be in directory data: eg ./data/II6-B2/*.iv
 - start with ipython -i iv.py <name of the data directory>
   - e.g. ipython -i iv.py II6-B2
 - automatically converts all *.iv text files in the given directory to a single hdf5 file 
 
## Usage
 - creates varialbe z with the class instance
 - plotting:
   - ```z.draw()```, ```z.draw_time()```, etc.
 - type ```z.draw?``` for more information
 - give argument "file_name" to save the plot as pdf to save plot in results folder
   - ```z.draw(file_name='bla')```
 - type save_last() to save the last canvas in the main directory 
   - ```save_last()```
