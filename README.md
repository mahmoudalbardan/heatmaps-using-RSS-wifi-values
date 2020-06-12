# heatmaps
Script to compute the heatmap.
Heatmaps are saved in the folder "./figures". 
"./json_files" is a temporary file where data are copied, unziped and treated to produce dataframes that are pickled as pandas.DataFrame objects in "./pickled_dataframes". These dataframes are used to compute the heatmaps.


Run using:


`
python3 heatmap.py [--directory DIRECTORY] [--location LOCATION] [--startdate STARTDATE] [--period PERIOD] [--edge_size EDGE_SIZE] [--sigma SIGMA] [--hcoef HCOEF] [--offset OFFSET]`

directory: directory of json files

location: name of the pharmacy. Only four location are supported for the moment ["fontenelle","canuts","colmar","venissieux"]

startdate: string in the format "%Y-%M-%d" starting date of the interval for which we want to build our heatmap

period: int, size of the interval 

edge_size: float [0.01:0.99], edge size for generating mesh

sigma: int, parameter of the Gaussian filter for the heatmap

hcoef: float, weight used to determine the extreme value of the heatmap

offset: int, offset applied when plotting and used for limit specification

