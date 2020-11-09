"""
Script to compute the heatmap
"""

import pandas as pd
import json
import pymysql
import numpy as np
from collections import Counter
from sklearn.feature_extraction import DictVectorizer
import os
from sklearn.cluster import KMeans
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import dmsh
import argparse
from shutil import copy
from zipfile import ZipFile

def get_mesh(area,contour):
    """
    get the points resulting from meshing
    
    Parameters:
    ----------
    area: int, area of small zones
    contour:list of tuples containing contour points coordinates
    
    Return:
    ------
    points: numpy.array, coordinates of the points forming the vertices of the mesh
    """
    contour = np.array(contour)/100
    esize = area
    geo = dmsh.Polygon(contour)
    vertices, cells = dmsh.generate(geo, esize)
    vertices = (vertices*100).astype(int)
    points = []
    for cell in cells:
        i,j,k = cell[0],cell[1],cell[2]
        point = (vertices[i]+ vertices[j]+ vertices[k])/3
        points.append(point.astype(int))
    points = np.vstack((points))
    
    return points,vertices,cells



def read_json(path_to_dbdict):
    """
    Read json file to get db_dictionary
    """
    with open(path_to_dbdict) as json_file:                
        db_dictionary = json.load(json_file)  
    return db_dictionary 
    
    
    
def build_connection(path_to_dbdict,location):
    """ 
    Establish connection to MySQL 
    
    Parameters:
    ----------
    db_name: string, database name where it looks for data_intrusion
    
    Return:
    ------
    connection: pymysql.connect  
    """
    db_dictionary = read_json(path_to_dbdict)
    credentials = db_dictionary[location]
    connection =  pymysql.connect(host=credentials['host'],
                         user = credentials['user'],
                         password = credentials['password'] ,
                         db=credentials['name'],
                         cursorclass=pymysql.cursors.DictCursor) 


    return connection

def get_positions(path_to_dbdict,location):
    """    
    Parameters:
    ----------
    path_to_dbdict: string, path to a local database dictionary to get sensor positions
    location: string, location of the drugstore ex: canuts, cusset,..
    
    Return:
    ------
    dfsensors:pd.DataFrame, sensors coordinates
    """     
    db_dictionary = read_json(path_to_dbdict)
    connection  = build_connection(path_to_dbdict,location)   
    dbname = 'fdv_hno' if location=='hno' else 'fdv_sanofi'            
    zone_id = db_dictionary[location]['zone_id']
    cursor = connection.cursor()
    query =  """SELECT LOWER(mac_address),position_x,position_y FROM {dbname}.app_devices 
                WHERE commercial_zones={zone_id}""".format(dbname=dbname,zone_id=zone_id)

    cursor.execute(query)
    data = cursor.fetchall()
    connection.close()
    dfsensors = pd.DataFrame(data)
    dfsensors.columns = ['sensor','position_x','position_y']
    return dfsensors



def remove_adds(df):
    """
    remove adds that stays more than timestay
    
    Parameters:
    ----------
    df: pd.DataFrame, describing data
    timestay: int, amount of time above which we consider that the mac is not a visitor
    """ 
    timestay = 3600
    gsf_add = df.groupby([pd.Grouper(key='address')])
    gps_add = [g for g in gsf_add ]    
    gps_add = [g for g in gsf_add if abs(float(min(pd.to_numeric(g[1]['date'], errors='coerce')))
               - float(max(pd.to_numeric(g[1]['date2'], errors='coerce'))))<timestay]
    regrp = [g[1] for g in gps_add]
    regdf = pd.concat(regrp)
    return regdf

    
def get_df(location,date,dir_):
    """
    get dataframe using the location and date from pickled files in dir   
    to be plugged in preprocessing
    
    Parameters:
    ----------
    dir_: string, path to the pickled dataframes of the data from the database
    date: string, date for which we want to compute the heatmap
    location: string, location for which we compute the heatmap ex: canuts,..
    
    Return:
    ------
    gps2: list of pd.Grouper, grouped by timestamp
    """
    filename = 'df-'+location+'-'+date+'.pkl'
    df = pd.read_pickle(os.path.join(dir_,filename))
    
    # CAN BE USED TO SPEFICY SOME SENSORS
    #sensorlist = ['453593','45354b','45356a']
    #df = df[df['device_mac_address'].isin(sensorlist)]
    
    
    df = remove_adds(df)  # timestay 3600
    
    df['date'] = pd.to_datetime(df['date'],unit='s')
    df = df.sort_values(by='date')    
    df['rssi'] = df['rssi'].astype('int')
     
    gsf1 = df.groupby([pd.Grouper(key='date',freq='24H')])
    gps1 = [g for g in gsf1]
    try:
        df1 = gps1[-1][1]        
        gsf2 = df1.groupby([pd.Grouper(key='date')])
        gps2 = [g for g in gsf2]
        return gps2
    except:
        return []
    

def handle_hist(hist_features):
    """
    Function that transform dict to array of features
    
    Parameters:
    ----------
    hist_features: list of dicts, resulting from self.histogram_features
    
    Return:
    ------
    array_hist: array of shape=(N,number of bins )
        
    """

    v = DictVectorizer(sparse=False) 
    dicted_vec = v.fit_transform(hist_features).flatten('C') 
    dicts = np.array(np.split(dicted_vec, len(hist_features)))
    dicts_df = pd.DataFrame(dicts)
    dicts_df = dicts_df.loc[:, (dicts_df != 0).any(axis=0)]  
    array_hist = np.array(dicts_df)
    feature_names = v.get_feature_names()
    return feature_names,array_hist


def preprocessing(gps):
    """
    preprocessing data: compute an array to cluster RSS values
    
    Parameters:
    -----------
    gps: list of pd.Grouper by timestamp
    
    Return:
    ------
    prep_data: numpy.array, array of shape n*m where n: address macs and m: wifi sensors 
    prep_time: numpy.array, array of shape 1*n where n: timestamps = address macs
    feature_names: list of strings, sensor list name where the order corresponds to RSS ordering in prep_data   
    """
    list_of_dicts = []
    times = []
    for gp in gps:
        if 7<=gp[0].hour<=22:
            times.append(gp[0])
            array = gp[1][['device_mac_address','rssi']].values
            dict_ = {}
            for row in array:
                dict_[row[0]]=row[1]
            list_of_dicts.append(dict_)
    feature_names,features = handle_hist(list_of_dicts)
    bools = ~np.any(features <= -80, axis=1) & ~np.any(features ==0, axis=1)
    prep_data = features[bools]
    prep_time = np.array(times)[bools]
    return prep_data, prep_time, feature_names
    
def interval_preprocessing(location,dates,dir_):
    """
    preprocessing multiple dates with the same location
    and stack their data verticaly
    Parameters:
    ----------
    location: string, location of the drugstore
    date: list (or array) of string, dates for which we want to compute the heatmap
    dir_: directory containing pickled dataframes containing source data
        
    Return:
    ------
    data: list of arrays, containing rss values to be clustered later (output of the function processing)
    time: list of timestamps,
    list_of_sensors: list of sensors, in order of the rssi values in the arrays in data
    """
    data,time,list_of_sensors,dfs = [],[],[],[]
    for date in dates:
        try:
            df = get_df(location,date,dir_)
            data.append(preprocessing(df)[0])
            time.append(preprocessing(df)[1])
            list_of_sensors.append(preprocessing(df)[2])
            dfs.append(df)
        except:
            continue
    return data, time, list_of_sensors,dfs
    
def clustering(n_clusters,data):
    """
    function that perform clustering of the data using KMeans algorithm
    
    Parameters:
    ----------
    n_clusters: int, number of clusters defined by the meshing
    data: numpy.array, data to be clustered (output of preprocessing function)
    
    Return:
    ------
    clusters: numpy.array, cluster centers equal to the n_clusters
    predictions: numpy.array, predictions of each sample (same length as data)
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    predictions = kmeans.predict(data)
    clusters = kmeans.cluster_centers_.astype('int')
    return clusters, predictions
    

def preparing_orderingdata(list_data,list_time,list_sensors,dfsensors,n_clusters):
    """
    Function that prepare preprocessed data (output of the preprocessing function)
    to following step of ordering clusters and points for each sensor
    
    Parameters:
    ----------
    list_data: list of data arrays for each data, to be vstacked
    list_time: list fo timestamps,
    list_sensors: list of sensors in the same order as clusters
    df_sensors:pd.DataFrame, positions of the sensors with their names
    n_clusters:int,number of clusters 
        
    Return:
    ------
    positions: numpy.array, positions of the sensors (in the same order as clusters)
    clusters: numpt.array array of shape n*m: n number of clusters and m: number of sensors
    predictions: numpy.array, array of shape 1*n of sample predictions
    """
    dims = []
    for d in list_data:
        dims.append(d.shape[1])
    dim = most_frequent(dims)
    data = [d for d in list_data if d.shape[1]==dim]
    data = np.vstack((data))  
    time = [t for sublist in list_time for t in sublist]
    sensors = list_sensors[0]
    dfsensors.columns = ['sensor','position_x','position_y']
    positions =  dfsensors.set_index('sensor').loc[sensors].reset_index(inplace=False)[['position_x','position_y']].values.astype(int)
    clusters,predictions = clustering(n_clusters,data)
    return time, positions, clusters, predictions


def ordering_points(to_keep,positions,points):
    """
    Ordering the points from the closest to the farest for each sensor
    
    Parameters:
    ----------
    to_keep: float, percentage of points to take into
    consideration for each sensor when ranking 
    the meshing points from the closest to the farest (plus petit des majorants tel que tout les points sont couvert)
    positions: numpy.array, positions of sensors
    points: numpy.array, cells positions (computed from meshing)
    
    Return:
    -------
    dict_ranking: dict, keys are sensor indexes and values are the points indexes
    """ 
    #to_keep = 0.4 for fontenelle with 46 cells
    dict_ranking = {}
    for j,sensor_position in enumerate(positions):
        dists = []
        for point in points:
            a,b = point,sensor_position
            dist = np.linalg.norm(a-b)
            dists.append(dist)
        closest = sorted(range(len(dists)), key=lambda k: dists[k])
        closest = closest[:int(to_keep*len(closest))+1]
        dict_ranking['sensor_'+str(j)]=closest
        
    return dict_ranking

def ordering_clusters(to_keep,positions,clusters):
    """
    Ordering the clusters from the closest to the farest for each
    sensor according to their RSS values
    
    Parameters:
    ----------
    to_keep: float, percentage of clusters to take into
    consideration for each sensor when ranking 
    the meshing points from the closest to the farest (plus petit des majorants tel que tout les points sont couvert)
    positions: numpy.array, positions of sensors
    clusters: numpy.array, RSS values of the cluster (size equal to the number of sensors)
    
    Return:
    -------
    dict_ranking: dict, keys are sensor indexes and values are the clusters indexes
    """ 
    allsensors = []
    for sensor_id in range(len(positions)):
        onesensor = []
        for j,cl in enumerate(clusters):
            if cl.tolist().index(np.max(cl))==sensor_id:
                onesensor.append([j,cl[sensor_id]])
        allsensors.append(np.vstack(onesensor))
        
    dict_ranking ={}   
    for j,array in enumerate(allsensors):
        idxs = sorted(range(len(clusters[:,j])), key=lambda k: clusters[:,j][k])[::-1]
        idxs = idxs[:int(to_keep*len(idxs))+1]
        dict_ranking['sensor_'+str(j)] =  idxs
        
    return dict_ranking


def compute_clusters_ratios(predictions):
    """
    Compute the ratio of each clusters in the predictions (all data)
    
    Parameters:
    ----------
    predictions: numpy.array, array of shape 1*n of sample predictions
    
    Returns:
    -------
    clusters_ratios: dictionary, keys are clusters indexes and values are ratios
    """
    clusters_countings = {k: v for k, v in sorted(dict(Counter(predictions)).items(), key=lambda item: item[1])}
    clusters_ratios={}
    for key in list(clusters_countings.keys()):
        clusters_ratios['cluster_'+str(key)] = clusters_countings[key]/len(predictions)    
        
    return clusters_ratios


def replace_closest(a,points):
    """
    if a point doesnt have a value, give it the value of the closest point
    """
    for j,ele in enumerate(a):
        if ele is None:
            dists = []
            for k,pt in enumerate(points):
                dist = np.linalg.norm(points[j]-points[k])
                dists.append(dist)
            sort_index = np.argsort(sorted(dists))
            for st in sort_index:
                if a[st] is not None:
                    break
            a[j] = a[st]
    return a

    
def cluster2point(centers,pts,points):
    """
    Function that links  clusters of rssi to points in the plan
    
    Parameters:
    ----------
    centers: numpy.array, ordered_clusters dictionary to array
    pts:numpy.array, ordered_points dictionary to array
    points:  numpy.array, points of clusters (centers of clusters)
    
    Return:
    ------
    cl2pt: numpy.array of shape (n_clusters,2) where first column is cluster number 
    and second column is point index
    """
    n_clusters = points.shape[0]
    cl2pt= []
    for c in range(n_clusters):
        x = True
        _i_ = []
        for j,row in enumerate(centers):
            try:
                _i_.append(row.tolist().index(c))
            except:
                _i_.append(np.inf)
            
        j_ = _i_.index(min(_i_))
        try:
            if c!=0:
                if pts[j_,min(_i_)] not in np.vstack((cl2pt))[:,1]:
                    cl2pt.append([c,pts[j_,min(_i_)]])
                    x= False
                if pts[j_,min(_i_)] in np.vstack((cl2pt))[:,1] and x==True:
                    j_ = _i_.index(sorted(_i_)[1])     
                    cl2pt.append([c,pts[j_,sorted(_i_)[1]]])     
            if c==0:
                cl2pt.append([c,pts[j_,min(_i_)]])
        except:
            continue
    return cl2pt


def point2cluster(cl2pt,points):
    """
    Function that links points in the plan to  of rssi clusters
    
    Parameter:
    --------
    cl2pt:numpy.array of shape (n_clusters,2) where first column is cluster number 
    and second column is point index (output of the function cluster2point)
    points: numpy.array, points of clusters (centers of clusters)
    
    Return:
    ------
    list_cls: list of ints, list of length n_clusters 
    """
    n_clusters = points.shape[0]
    _cls_ = []  
    cl2pt = np.vstack((cl2pt))      
    for pt in range(n_clusters):
        id_ = np.where(cl2pt[:,1]==pt)[0]
        _cls_.append(cl2pt[id_,0])
             
    list_cls = []   
    for _cl_ in _cls_:
        try:
           list_cls.append(_cl_[0])
        except:
           list_cls.append(None)
    
    list_cls = replace_closest(list_cls,points)
    return list_cls
    
def compute_heatmap1d(predictions,list_cls):
    """
    Construct heatmap vector of length equal to n_clusters
    
    Parameter:
    ---------
    predictions: numpy.array, array of shape 1*n of sample predictions
    list_cls: list of ints, list of cluster indexes
        
    Return:
    ------
    heatmap: list of floats, ratios of each zone to the whole plan
    """
    clusters_ratios = compute_clusters_ratios(predictions)
    heatmap_1d = []        
    for ele in list_cls:
        try:
            if ele is not None:
               heatmap_1d.append(clusters_ratios['cluster_'+str(ele)])
            if ele is None:
               heatmap_1d.append(0)
        except:
            heatmap_1d.append(0)
    return heatmap_1d

def heatmap_conditions(offset,points):
    """
    Determine some heatmap conditions while plotting
    
    Parameter:
    ---------
    offset: int, offset to be applied when plotting
    points: numpy.array, points of clusters (centers of clusters)
    
    Return:
    ------
    (size_x,size_y): tuple, shape of the array that will contain the heatmap
    xlim, ylim: tuples of ints, interval limits on each axis
    """ 
    size_x,size_y = np.max(points[:,0])+offset, np.max(points[:,1])+offset
    xlim = (np.min(points[:,0])-offset,np.max(points[:,0])+offset)
    ylim = (np.min(points[:,1])-offset,np.max(points[:,1])+offset)
    return size_x, size_y, xlim, ylim[::-1]

def compute_heatmap2d(predictions,list_cls,offset,points):
    """
    Construct heatmap vector in 2D
    
    Parameter:
    ---------
    predictions: numpy.array, array of shape 1*n of sample predictions
    list_cls: list of ints, list of cluster indexes
    offset: int, offset to be applied when plotting heatmap
    points: numpy.array, points of clusters (centers of clusters)
        
    Return:
    ------
    heatmap_2d: list of floats, ratios of each zone to the whole plan
    """
    heatmap_1d= compute_heatmap1d(predictions,list_cls)
    size_x,size_y=heatmap_conditions(offset,points)[0],heatmap_conditions(offset,points)[1]
    heatmap_2d = np.zeros(shape=(size_x,size_y))
    for k,pt in enumerate(points):
        i,j = pt[0]+20, pt[1]+20
        heatmap_2d[i,j] = heatmap_1d[k]*len(predictions)
    return heatmap_2d
            

def plot_heatmap(figsdir,
                 location,
                 dates,
                 heatmap_2d,
                 points,
                 vertices,
                 cells,
                 sigma,
                 hcoef,
                 dim,
                 offset,
                 cmap):
    """
    Plot the heatmap
    
    Parameters:
    ----------
    heatmap_2d:numpy.array of floats, 2d array of the heatmap
    vertices:
    cells:
    sigma:int, parameter of the gaussian filter 
    dim:tuple of int, dimension of the figure
    offset: int, offset applied when plotting and used for limit specification
    cmap: string, color map
    figsdir: string, directory to save figure
    location: string, location name
    dates: list of strings, list of dates
    """
    xlim,ylim=heatmap_conditions(offset,points)[2],heatmap_conditions(offset,points)[3]
    fig, axes= plt.subplots(ncols=1, nrows=1,figsize=dim)
    plt.triplot(vertices[:, 0]+20, vertices[:, 1]+20, cells)
    heatmapext = np.max(gaussian_filter(heatmap_2d, sigma=sigma))*hcoef
    smoothed_heatmap = gaussian_filter(heatmap_2d, sigma=sigma)
    hm =  sns.heatmap(smoothed_heatmap.T, 
                      vmin=-heatmapext,
                      vmax=heatmapext,
                      cmap =cmap,
                      ax=axes) 
    
    hm.set(ylim=ylim)
    hm.set(xlim=xlim)
    
    figname = location + '_' + dates[0] + '_' + dates[-1] 
    figpath = os.path.join(figsdir,figname)
    fig = hm.get_figure()
    fig.savefig(figpath) 
    


def mainfunction(dfs_dir,dfsensors,points,location,startdate,period,n_clusters,offset,vertices):
    """
    Function that perform the necessary calculation to produce the heatmap
    if we have a problem, the while loop is will try until it is fixed
    
    Parameters:
    ----------
    dfs_dir: string, path to pickled dataframes
    dfsensors: dataframe, positions of the sensors
    location: string, name of the drugstore
    startdate: string, starting date in %Y-%m-%d format
    period:int, the period for which heatmap is computed 
    n_clusters: int, number of clusters
    offset:int, offset applied to be padded in the heatmap 2d-array
        
    Return:
    ------
    centers:numpy.array of shape (n_sensors,n_clusters) with ordering
    pts: numpy.array of shape (n_sensors,n_clusters=n_points) with ordering
    """
    dates =  pd.date_range(start=startdate, periods=period).strftime("%Y-%m-%d")  
    dates_mod = dates
    i=0
    while True:
        try:
            data,time,list_of_sensors,dfs = interval_preprocessing(location,dates_mod,dfs_dir)
            time, positions, clusters, predictions = preparing_orderingdata(data,time,list_of_sensors,dfsensors,n_clusters)
            ordered_points = ordering_points(0.999,positions,points)
            ordered_clusters = ordering_clusters(0.999,positions,clusters)        
            centers = list(ordered_clusters.values())
            centers = np.vstack((centers))
            pts = list(ordered_points.values())
            break
        except:
            dates_mod = np.random.choice(dates, len(dates)-1)
            continue
        break
        i+=1
    pts = np.vstack((pts))  
    cl2pt = cluster2point(centers,pts,points)
    list_cls=  point2cluster(cl2pt,points) 
    heatmap_2d = compute_heatmap2d(predictions,list_cls,offset,points)
    return heatmap_2d



def most_frequent(List): 
    return max(set(List), key = List.count) 


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        
        
def read_file_fromlocal(file):
    """
    read the files after being downloaded from the server
    """
    if file.endswith('.json'):
        with open(file) as json_file:                
            loaded_json = json.load(json_file)   
    return loaded_json

def save_pkldf(date,location,location_sensors,filematch,directory_json,directory_pkl):
    """
    Create a dataframe from json files corresponding to one day
    
    Parameters:
    ----------
    date: string, date in format %Y-%m-%d e.g. 2020-02-18
    location:string , location
    location_sensor:
    filematch:
    directory_json: directory of json files
    directory_pkl: diretory to which save pkl files
    
    
    Return:
    -------
    df: pd.DataFrame, datadframe constructed from json files for a specific day and location        
    """
    list_of_dataframes = []
    contents = [os.path.join(directory_json,file) for file in os.listdir(directory_json) if filematch in file and date in file]
    for j,file in enumerate(contents):
        json_file = read_file_fromlocal(file)
        df = pd.DataFrame(json_file)
        rssi,addresses = [],[]
        for i,row in df.iterrows():
            rssi = rssi + df.iloc[i].values[0]['rssi']
            to_add = np.repeat(df.iloc[i].values[0]['address'],len(df.iloc[i].values[0]['rssi'])).tolist()
            addresses = addresses + to_add
        df1 = pd.DataFrame(addresses)
        df2 = pd.DataFrame(rssi)
        frames = [df1,df2]
        df = pd.concat(frames,axis=1)
        df.columns.values[0] = "address"  
        sensors = list(location_sensors['capteurs'][location_sensors['location']==location])[0]            
        sensors = [ele.lower() for ele in sensors]
        list_of_dataframes.append(df.loc[df['device_mac_address'].isin(sensors)])
    df = pd.concat(list_of_dataframes)
    name = 'df'+'-' +location +'-' +date+'.pkl'
    path = os.path.join(directory_pkl,name)
    df.to_pickle(path)
    

def copy_unzip(source_directory,directory_json,dates):
    """
    Parameters:
    ----------
    source_directory: string, source directory where we can find json files
    directory_json: string, path to intermediate directory to copy to
    dates: list of strings, dates

    """
    months = []
    for ele in dates:
        months.append(ele[:7])   
    months = list(Counter(months).keys())
    paths = [os.path.join(source_directory,file) for file in os.listdir(source_directory) if any(substring in file for substring in months)]
    for path in paths:
        copy(path, directory_json)
    
    list_of_files = [os.path.join(directory_json,file) for file in os.listdir(directory_json) if file.endswith('zip') or file.endswith('json')]
    for file in list_of_files:
        if file.endswith('.zip'):
            with ZipFile(file, 'r') as zipfile:
                zipfile.extractall(directory_json)
                
                
def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=str,
                        help='source directory of the json files', default='./var/datas-json')

    parser.add_argument("--location", type=str,
                        help='name of the pharmacie', default='fontenelle')
    parser.add_argument('--startdate',type=str,
                        help='starting date in the format %Y-%m-%d', default='2020-02-21')
    parser.add_argument('--period',type=int,
                        help='size of the interval for which we want to compute the heatmap', default=10)
    parser.add_argument('--edge_size',type=float,
                        help='edge size for meshing', default=0.5)
    parser.add_argument('--sigma',type=int,
                        help='parameter for the gaussian filter', default=8)
    parser.add_argument('--hcoef',type=float,
                        help='heatmap coefficient to determine xtrem values', default=1)
    parser.add_argument('--offset',type=int,
                        help='offset to plot the heatmap', default=50)
    args = parser.parse_args()
    return args


       
def main():
    
    figsdir = './figures' # directory of the figures
    directory_pkl = "./pickled_dataframes" # pickled dataframes directory
    directory_json = "./json_files" # intermediate json directory (temporary)
    print (figsdir)
    
    args = parse_args()
    source_directory = args.directory # source json directory
    location = args.location
    startdate = args.startdate
    period = args.period
    edge_size = args.edge_size
    sigma = args.sigma
    hcoef = args.hcoef
    offset = args.offset
    
    filematch = '.json'
    df_sensors = pd.read_pickle("./sensors")[location].dropna()
    contour = pd.read_pickle("./contours")[location].dropna()
    location_sensors = pd.read_pickle("./location_sensors")
    dates =  pd.date_range(start=startdate, periods=period).strftime("%Y-%m-%d")  
    #print (location_sensors.location,dates,contour,df_sensors)
    
    # copy data to other folder and unzip when necessary
    print ("copy begins")
    copy_unzip(source_directory,
               directory_json,
               dates)
    print ("copy ends")
    # change porte des alpes location sensors
    print ("pickled saving begins")
    for date in dates:
        save_pkldf(date,
                   location,
                   location_sensors,
                   filematch,
                   directory_json,
                   directory_pkl)
    print ("pickled saving ends")
    
    points,vertices,cells = get_mesh(edge_size,contour)
    n_clusters = points.shape[0]
    print (n_clusters)
    heatmap_2d = mainfunction(directory_pkl,
                              df_sensors,
                              points,
                              location,
                              startdate,
                              period,
                              n_clusters,
                              offset,
                              vertices)
    
    plot_heatmap(figsdir,
                 location,
                 dates,
                 heatmap_2d,
                 points,
                 vertices,
                 cells,
                 sigma =sigma,
                 hcoef=hcoef,
                 dim = (7, 4),
                 offset=offset,
                 cmap='nipy_spectral')
    




if __name__ == "__main__":
    main()
