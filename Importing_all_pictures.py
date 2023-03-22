import pandas as pd
import numpy as np
import dash                     #(version 1.0.0)
from dash import dcc
from dash import html
import plotly.express as px
import plotly.graph_objs as go
import dash_bootstrap_components as dbc
import os

""" Importing Data """
path_to_all_data = r"C:\Users\caspe\OneDrive - TU Eindhoven\DS&AI 2022-2023\Q3\2AMV10 - Visual Analytics\2. Terrorists at CGCS\data\data"


from PIL import Image
#importing the Image class from PIL package
def picture_import(path):
    person_images = { }
    
    for subdir, dirs, files in os.walk(path_to_all_data):
        person = subdir[116:] #This is different for everyone as folders are named differently!!
        person_images[person] = []
        for file in files:
            im_path = os.path.join(subdir, file)
            str_im_path = str(im_path)
            if str_im_path[-1] == 'g':
                
                #read the image, creating an object
                im = Image.open(im_path)
                person_images[person].append(im)
            
    return person_images


picture_dict = picture_import(path_to_all_data)
