# heatmaps
A new method to compute the localization heatmap using RSS values of wifi signals emitted from mobile phones. 
We installed wifi sensors in multiple drugstores and tried to construct heatmaps of people position inside them. 
The method is based on signal strengh. 
Instead of classical trilateration (which is not suitable in a closed environment), we compute the heatmap based on clustered RSS.
If we have a mobile phone *p* and *n* sensors inside a store, we are able to capture *n* signals, then we have *n*  RSS values
vector = (*p_1*, *p_2*, ..., *p_n*). Now, if we have 1000 clients in the store, we can construct a 2D dataset of size 1000 * *n* where the cell (i,j) in this dataset correspond to the RSS value of the signal emitted from the mobile phone *i* and captured by the sensor *j* .


**Important**:   
Heatmaps are saved in the folder `./figures`.           
`./json_files` is a file where data are saved temporarly, copied, unziped and treated to produce dataframes that are pickled as pandas.DataFrame objects in `./pickled_dataframes`.         
These dataframes are used to compute the heatmaps.      

**Run using**:       
Run your code using 
` python3 heatmap.py [--directory DIRECTORY] [--location LOCATION] [--startdate STARTDATE] [--period PERIOD] [--edge_size EDGE_SIZE] [--sigma SIGMA] [--hcoef HCOEF] [--offset OFFSET]`

**Arguments**:

`directory:` directory of json files        
`location`: name of the pharmacy. Only four location are supported for the moment [`fontenelle`,`canuts`,`colmar`,`venissieux`]         
`startdate`: string in the format "%Y-%M-%d" starting date of the interval for which we want to build our heatmap          
`period`: int, size of the interval. Default to 30 (over one month)       
`edge_size`: float [0.01:0.99], edge size for generating mesh. Default to 0.5          
`sigma`: int, parameter of the Gaussian filter for the heatmap. Default to 10           
`hcoef`: float, weight used to determine the extreme value of the heatmap. Default to 1.05         
`offset`: int, offset applied when plotting and used for limit specification.Default to 50      

