# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 11:53:07 2023

@author: caspe
"""
import numpy as np
import pandas as pd
from collections import Counter
import sys
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import normalize
import os 

predictions_df = pd.read_csv(r"C:\Users\caspe\Documents\GitHub\VisAnProject\predictions.csv")
path_to_all_data = r"C:\Users\caspe\OneDrive - TU Eindhoven\DS&AI 2022-2023\Q3\2AMV10 - Visual Analytics\2. Terrorists at CGCS\data\data"

from PIL import Image
#importing the Image class from PIL package
def picture_import(path):
    person_images = {}
    person_text = {}
    all_pictures = {}
    all_captions = {}
    for subdir, dirs, files in os.walk(path_to_all_data):
        person = subdir[116:] #This is different for everyone as folders are named differently!!
        person_images[person] = []
        person_text[person] = []

        for file in files:
            z = file
            im_path = os.path.join(subdir, file)
            str_im_path = str(im_path)
            if str_im_path[-1] == 'g':
                
                #read the image, creating an object
                im = Image.open(im_path)
                person_images[person].append((file,im))
                
                all_pictures[file] = im
            if str_im_path[-1] == 't':
                with open(str_im_path, 'r') as file:
                    data = file.read().replace('\n', '')
                    person_text[person].append((z, data))
                    all_captions[z] = data
                    
    return person_images, person_text, all_pictures, all_captions

data_import = picture_import(path_to_all_data)
picture_dict = data_import[0]
text_dict = data_import[1]
all_pictures = data_import[2]
all_captions =  data_import[3]
#%%
import itertools
from skimage.metrics import mean_squared_error
from skimage.transform import resize
from skimage.metrics import normalized_mutual_information
from skimage.metrics import normalized_root_mse

def ImPersonSim(Px, Py, nmi =False, norm_rmse = False, MSE = True ):
    images_Px = picture_dict[Px]
    images_Py = picture_dict[Py]
    
    list1 = [x[0] for x in images_Px]
    list2 = [x[0] for x in images_Py]

    list = [list1, list2]
    combinations = [p for p in itertools.product(*list)]
    results = {}
 
    if MSE: 
        MSE_dict = {}
        for comparison in combinations:
            picture_x = np.array(all_pictures[comparison[0]].convert('L'))
            picture_y = np.array(all_pictures[comparison[1]].convert('L'))
            
            picture_x = resize(picture_x, (3024, 4032))
            picture_y = resize(picture_y, (3024, 4032))
            
            name = comparison[0] + '_' + comparison[1] 
            iets =  mean_squared_error(picture_x,picture_y)
            print(name, ':', iets)
            MSE_dict[name] =iets 
            
        results['MSE'] = MSE_dict
    
    if nmi: 
        """ Normalized Mutual Information ranges from 1-2 where 1= totally uncorrelated and 2 = completely the same
        """
        nmi_dict = {}
        for comparison in combinations:
            picture_x = np.array(all_pictures[comparison[0]].convert('L'))
            picture_y = np.array(all_pictures[comparison[1]].convert('L'))
            name = comparison[0] + '_' + comparison[1] 
            iets =  normalized_mutual_information(picture_x,picture_y)
            print(iets)
            nmi_dict[name] =iets 
            
        results['NMI'] = nmi_dict
    if norm_rmse:
        norm_rmse_dict = {}
        for comparison in combinations:
            picture_x = np.array(all_pictures[comparison[0]].convert('L'))
            picture_y = np.array(all_pictures[comparison[1]].convert('L'))
            
            picture_x = resize(picture_x, (3024, 4032))
            picture_y = resize(picture_y, (3024, 4032))
            
            name = comparison[0] + '_' + comparison[1] 
            iets = normalized_root_mse(picture_x,picture_y)
            print(name, ':',iets)
            norm_rmse_dict[name] = normalized_root_mse(picture_x,picture_y)
        results['norm_mse'] = norm_rmse_dict
    return results

pers1 = 'Person27'
pers2 = 'Person7'
h = ImPersonSim(pers1, pers2)
picture_x = np.array(all_pictures['Person9_2.jpg'].convert('L'))
picture_y = np.array(all_pictures['Person8_4.jpg'].convert('L'))
#%%
picture_x = resize(picture_x, (3024, 4032))
picture_y = resize(picture_y, (3024, 4032))
print(mean_squared_error(picture_x,picture_y))











#%%
def sim_matrix():
    predictions_df_rank1 = predictions_df[predictions_df['rank']>= 1]

    alll = Counter(predictions_df_rank1['class_label'])
    allowed_classes = [k for k,v in alll.items() if v >= 10]
    
    predictions_df_rank2 = predictions_df_rank1[predictions_df_rank1['class_label'].isin(allowed_classes)]
    
    all_instances = np.unique(predictions_df_rank2['class_label'])
    
    
    
    dictt = {}
    for i in all_instances:
        dictt[i] = 0
    predictions_df_rank2.reset_index()
    persons = np.unique(predictions_df_rank2['person_id'])
    al_dicts = []
    matrix = []
    for i in persons:
        predictions_df_rank1_person = predictions_df_rank2[predictions_df_rank2['person_id']==i]
        count_dict = Counter(predictions_df_rank1_person['class_label'])
        
        for q in all_instances:
            if q not in count_dict.keys():
                count_dict[q] = 0
        count_dict = dict(sorted(count_dict.items()))
        count_dict['Saturarion'] = float(np.random.uniform(-100, 100, 1))
        count_dict['Stiffness'] = float(np.random.uniform(-100, 100, 1))
        count_dict['Polarity'] = float(np.random.uniform(-100, 100, 1))
        count_dict['Tone'] = float(np.random.uniform(-100, 100, 1))
        count_dict['Total_sentiment'] = float(np.random.uniform(-100, 100, 1))
        al_dicts.append(count_dict)
        
        matrix.append(list(count_dict.values()))

    X = np.array([np.array(xi) for xi in matrix])

    
    np.set_printoptions(threshold=sys.maxsize)
    
    X = normalize(X, axis=0, norm='l2')
    
    distance_matrix = pairwise_distances(X, metric='l1')
    distance_matrix = normalize(distance_matrix, axis=1, norm='l1')
    
    return distance_matrix


    

    
