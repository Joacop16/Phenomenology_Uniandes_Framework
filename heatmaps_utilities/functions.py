import pandas as pd
import seaborn as sns 
import numpy as np
from scipy.interpolate import griddata

import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "serif"

def read_excel(path, index_colum = 0):
    '''This function receives a path to an excel file and returns the DataFrame contained.
    
    Parameters:
    - path*: excel file path (file ending in .xlsx).
    - index_colum: column of the table that contains the indexes of the DataFrame (by default it is column 0).
    
    Return:
    - data: Pandas DataFrame.
    '''

    data = pd.read_excel(path, index_col = index_colum)
    data.sort_index(level=0, ascending=False, inplace=True)
    data.columns = [float(i) for i in data.columns]
    data.index = [float(i) for i in data.index]  
    
    return data

def read_csv(path, index_colum = 0):
    '''This function receives a path to an csv file and returns the DataFrame contained.
    
    Parameters:
    - path*: csv file path (file ending in .csv).
    - index_colum: column of the table that contains the indexes of the DataFrame (by default it is column 0).
    
    Return:
    - data: Pandas DataFrame.
    '''

    data = pd.read_csv(path, index_col = index_colum)
    data.sort_index(level=0, ascending=False, inplace=True)
    data.columns = [float(i) for i in data.columns]
    data.index = [float(i) for i in data.index]  
    
    return data

def smooth(Data, log = False):
    
    '''This function receives an array of data and interpolates the data to re-turn a larger and "continuous" DataFrame.
        
    Parameters:
    - Data*: DataFrame with the data, each entry refers to the value of a variable based on the column and row values. For example, if we consider two independent variables A, B and a function F(A,B), Data is a matrix where the rows take the values of A and the columns the values of B, i.e. each entry in the matrix will be just F(A,B).    
    - log: Boolean that says if we will work with logarithmic scale (base 10).

    Return:
    - Data_interpolated: Pandas DataFrame.
    '''
    
    index = Data.index #DataFrame row labels
    columns = Data.columns #DataFrame column labels
        
    if log: Data = np.log10(Data)
    
    #In order to apply griddata (scipy function that allows interpolation) it is necessary to rewrite the data of a matrix in a three-column format. Instead of having a matrix where the function is evaluated according to the row and column, it is better to rewrite everything in a three column format. Simply: A, B, F(A,B). This is what the following code does:    
    
    matrix = np.zeros([len(columns)*len(index),3])
    column_0 = []
    for i in range(len(columns)):
        for j in range(len(index)):
            column_0.append(columns[i])

    column_1 = []
    for j in range(len(columns)):
        for i in range(len(index)):
            column_1.append(index[i])

    matrix[:,0] = column_0
    matrix[:,1] = column_1

    for k in range(len(matrix[:,2])):
        matrix[k,2] = Data[matrix[k,0]][matrix[k,1]]

    Data = pd.DataFrame(matrix) 
    Data.columns = ["A", "B", "F(A,B)"] #Three-column format
    
    #At this point, the data can be easily read, now what must be done is to create the values ​​of x that we want to interpolate, that is, the continuous intervals that we want to interpolate, I will call these "x" and "y". However, the case in which one variable is fixed and only the other takes different values ​​must be taken into account, this is what the len(--) == 1 represent:

    if(len(columns) == 1): x = columns 
    else: x = np.linspace(np.min(Data['A']),np.max(Data['A']),500) 
    
    if(len(index) == 1): y = index 
    else: y = np.linspace(np.min(Data['B']),np.max(Data['B']),500) 
    
    #At this point we already have "x" and "y", now we must combine them to take into account all the combinations, this is done by np.meshgrid:   
    gridx, gridy = np.meshgrid(x,y)
    
    #Finally, it would be enough to interpolate using our data as a base; However, three cases must be taken into account:
    
    if (len(columns) == 1):
        Data_interpolated = griddata(Data['B'].values, Data['F(A,B)'].values, y, method='cubic') 
        Data_interpolated = pd.DataFrame(Data_interpolated)           
        Data_interpolated.index = y
        Data_interpolated.columns = x
        Data_interpolated.columns = [round(i,2) for i in Data_interpolated.columns]
        Data_interpolated.index = [round(i,2) for i in Data_interpolated.index] 
        
    elif (len(index) == 1):
        Data_interpolated = griddata(Data['A'].values, Data['F(A,B)'].values, x , method='cubic') 
        Data_interpolated = pd.DataFrame(Data_interpolated)
        Data_interpolated.index = x
        Data_interpolated.columns = y
        Data_interpolated = Data_interpolated.T #griddata always returns in the form of a column vector, but if in this case and it is fixed, our data must be in a row vector format, that does the .T, it takes out the transpose.
        Data_interpolated.columns = [round(i,2) for i in Data_interpolated.columns]
        Data_interpolated.index = [round(i,2) for i in Data_interpolated.index] 
        
    else:
        Data_interpolated = griddata((Data['A'].values,Data['B'].values), Data['F(A,B)'].values, (gridx,gridy), method='cubic')
        Data_interpolated = pd.DataFrame(Data_interpolated)
        Data_interpolated.index = y
        Data_interpolated.columns = x
            
    Data_interpolated.sort_index(level=0, ascending=False, inplace=True) #Allows the graph to show the y-axis growing upwards
    
    return Data_interpolated

def plot_heatmap(Data, level_curves = {}, level_curves_labels_locations = [], zoom_region = {}, **kwargs):

    '''This function plots the heat map of a DataFrame. In addition to this, it also plots contour lines and zoom_regions if the user wants it.
    
    Parameters:
    - level_curves: Directory containing the level curves to be ploted. It must have the structure {value (float): label(string),...}.
    - level_curves_labels_locations: List with suggested coordinates [(x1,y1), (x2,y2),...] of the positions where you want the signs of each level curve to be: The order does not matter.
    - zoom_region: Directory containing the positions to be zoom_regioned. It must have the structure {x1 (string): value (float), x2 (string): value (float), y1 (string): value (float), y2 (string): value (float),}.
    - **kwargs: Optional parameters like title, title_right, title_left, x_label, y_label, cbar_label, color, File_name, etc.
    
    Return:
    - fig, ax, curves: matplotlib.pyplot subplots and contours.
    '''
    
    curves = 0
    fig, ax = plt.subplots()
    
    try: plt.title(kwargs['title'])
    except: pass

    try: plt.title(kwargs['title_right'], loc = 'right')
    except: pass

    try: plt.title(kwargs['title_left'], loc = 'left')
    except: pass

    try: x_label = kwargs['x_label']
    except: x_label = ''

    try: y_label = kwargs['y_label']
    except: y_label = ''

    try: cbar_label = kwargs['cbar_label']
    except: cbar_label = ''
    
    try: color = kwargs['color']
    except: color = 'viridis'
    
    try: File_name = kwargs['File_name']
    except: File_name = ''
    
    
    
    index = Data.index
    columns = Data.columns
    
    if(len(index) == 1 or len(columns) == 1):
        Data.sort_index(level=0, ascending=False, inplace=True) 
        sns.heatmap(Data, cmap = color, cbar_kws={'label': cbar_label}).set(xlabel= x_label, ylabel= y_label)
        
    else:
        mapa_calor = plt.pcolormesh(columns, index, Data.values, cmap = color)
        plt.colorbar(mapa_calor, label = cbar_label)  
    
        plt.xlabel(x_label)
        plt.ylabel(y_label)
    
    #Depending on the shape of the data, the curves are plotted differently. In the event that one of the two independent variables is fixed, then the matplotlib contour function cannot be used due to the dimension of said variable. In that case, it is necessary to plot asymptotes .
    
    if (len(list(level_curves.keys())) != 0):

        index = Data.index
        columns = Data.columns

        if(len(index) == 1):
            linestyles = ['-', '--', '-.', ':', '']
            for i in range(len(list(level_curves.keys()))):
                curva = list(level_curves.keys())[i]
                indice = np.abs(np.asarray(Data - curva)).argmin()
                
                #It is plotted with the sign inside the plot:
                ax.axvline(indice, ls = linestyles[1], color = 'white') 
                ax.text(indice - 20, 0.5, level_curves[curva], rotation = 90, color = 'white')
                
                #It is plotted with the sign in a legend
                #ax.axvline(indice, label = level_curves[curva], ls = linestyles[i % 5], color = 'white') 
            # ax.legend()

        elif(len(columns) == 1):
            
            linestyles = ['-', '--', '-.', ':', '']
            for i in range(len(list(level_curves.keys()))):
                curva = list(level_curves.keys())[i]
                indice = np.abs(np.asarray(Data - curva)).argmin()
                
                #It is plotted with the sign inside the plot:
                ax.axhline(indice, ls = linestyles[1], color = 'white') 
                ax.text(0.5, indice + 20 , level_curves[curva], color = 'white')
                
                #It is plotted with the sign in a legend
                #ax.axhline(indice, label = level_curves[curva], ls = linestyles[i % 5], color = 'white')
            #ax.legend()
            
        else:
            curves = plt.contour(Data.columns, Data.index, Data.values, levels = list(sorted(level_curves.keys())), colors = ['white'], linestyles = 'dashed') #Level curves
            if (len(level_curves_labels_locations) != 0): ax.clabel(curves, curves.levels, manual = level_curves_labels_locations, fmt = level_curves, fontsize=10, rightside_up = True) #Labels level curves location
            else: ax.clabel(curves, curves.levels, inline=True, fmt = level_curves, fontsize=10, rightside_up = True) #Labels level curves

    if (len(list(zoom_region.keys())) != 0):
        
        if (len(index) == 1 or len(columns) == 1): print('It is not possible to zoom_region due to the dimensions of the DataFrame.')
        
        else:
        
            ax_zoom = ax.inset_axes([1.4, 0, 1, 1])
            
            ax_zoom.pcolormesh(Data.columns, Data.index, Data.values, cmap = color)
            
            if (len(list(level_curves.keys())) != 0): ax_zoom.contour(Data.columns, Data.index, Data.values, levels = list(sorted(level_curves.keys())), colors = ['white'], linestyles = 'dashed') #Curvas de nivel

            ax_zoom.set_xlim(zoom_region['x1'], zoom_region['x2'])
            ax_zoom.set_ylim(zoom_region['y1'], zoom_region['y2'])
            ax_zoom.set_xlabel(x_label)
            ax_zoom.set_ylabel(y_label)

            ax.indicate_inset_zoom(ax_zoom, edgecolor= "black")
            
    if (File_name != ''): plt.savefig(File_name, bbox_inches='tight')
    
    return fig, ax, curves