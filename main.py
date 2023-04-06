# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 14:40:46 2023

@author: caspe
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 20:44:17 2022

@author: caspe
"""
"""Importing Libraries
"""

import dash                     #(version 1.0.0)
from dash import dcc, html, dash_table, Input, Output,  callback
import dash_bootstrap_components as dbc
import os
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import pairwise_distances
import sys
import pandas as pd
from PIL import Image
import numpy as np
import plotly.express as px
from dash.exceptions import PreventUpdate
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from collections import Counter
import networkx as nx


""" Importing Data 
"""
path_to_all_data = r"C:\Users\caspe\OneDrive - TU Eindhoven\DS&AI 2022-2023\Q3\2AMV10 - Visual Analytics\2. Terrorists at CGCS\data\data"

predictions_df = pd.read_csv(r"C:\Users\caspe\Documents\GitHub\VisAnProject\predictions.csv")
#%%
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

"""Setting up data dictionaries
"""
data_import = picture_import(path_to_all_data)
picture_dict = data_import[0]
text_dict = data_import[1]
all_pictures = data_import[2]
all_captions =  data_import[3]

"""Making dataframe for every person
"""
persons = picture_dict.keys()
all_data_frames = {}
for i in persons:
    person_frame = pd.DataFrame(columns = ['Time' , 'Text', 'Picture'] )
    time = 0
    index = 0
    for pic_tup in picture_dict[i]:
        placed = False
        if text_dict[i]:
            
            for text_tup in text_dict[i]:
                if pic_tup[0][:-4] == text_tup[0][:-11]:
                    person_frame.loc[index] = [time, text_tup[1],pic_tup[0]]
                    time += 1
                    index += 1
                    placed = True
                    break
        elif not placed:
            person_frame.loc[index] = [time, 0,pic_tup[0]]
            time += 1
            index += 1
        else:
            person_frame.loc[index] = [time, 0,pic_tup[0]]
            time += 1
            index += 1
            
    all_data_frames[i]=person_frame


def sim_matrix(predictions_df): #text_analysis_dictionary
    """ Calculates the similarity between persons by building a matrix where columns represent various 
    features from the analysis and rows correspond to users. First n columns represent counts of how often 
    a particular item was recognized for a specific person in all their tweets using CNN Exception. The last 
    features are the resulting averaged Stiffness,Polarity, Tone over all tweets per person.
    
    To calculate the distance between persons pairwise_distances function from sci-kit is called. 
    
    Args:
        predictions_df: Pandas Dataframe containing picture id, classification, percentage 
        of certainty and rank of said classification.
        
        text_analysis_dictionary: Dictionary with key = "Personx" where x is the persons number 
        and a list of values in the following order [Stiffness,Polarity, Tone]

        Stiffness = the rate of "harsh/emotionless" words compared to the rate of "emotionful" words

        Polarity: The overall sentiment

        Tone = choice of words leading to interpretation and total amount of possible interpretations
        
        all in range (-100,100)

    Returns:
        Distance_matrix: A numpy matrix of size p times p, where p is the number of persons that are in the data. 
        Values in the matrix represent the distance between two persons, row and column respectively. 
        
        Smaller distances indicate a bigger similarity.
    """
    
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
        al_dicts.append(count_dict)
        
        matrix.append(list(count_dict.values()))

    X = np.array([np.array(xi) for xi in matrix])

    
    np.set_printoptions(threshold=sys.maxsize)
    
    X = normalize(X, axis=0, norm='l2')
    
    distance_matrix = pairwise_distances(X, metric='l1')
    
    
    return distance_matrix


sim_matrixx = np.round(sim_matrix(predictions_df),3)

def style_row_by_top_values(df, nlargest=15):
    numeric_columns = df.drop(['User'], axis=1).columns
    styles = []
    for i in range(len(df)):
        row = df.loc[i, numeric_columns].sort_values(ascending=False)
        for j in range(nlargest):
            styles.append({
                'if': {
                    'filter_query': '{{id}} = {}'.format(i),
                    'column_id': row.keys()[j]
                },
                'backgroundColor': '#39CCCC',
                'color': 'white'
            })
    return styles

def plot_network_graph(edge_list):
    """ Gets a list of edges to be constructed, constructs a graphical represention of network using 
    Networkx and then visualizes it using  plotly.graph_objects
    
    Args: 
        edge_list: list of edges in the following form [('Personx', 'Persony', edge_weight), () .....]
        where edge_weight = float or int
        

    Returns:
       a Go.Figure object that can be visualized in Dash
    """
    
    
    
    
    plt.figure(num=None, figsize=(10, 10), dpi=80)
    plt.axis('off')
    fig = plt.figure(1)
    weights = []
    
    G = nx.Graph()
    for edge in edge_list:
        if edge[2] >0:
            weights.append(str(edge[2]))
            
            G.add_edge(edge[0], edge[1], weight=edge[2])
    
    pos = nx.spring_layout(G)  # positions for all nodes - seed for reproducibility
    
    # edges trace
    edge_x = []
    edge_y = []
    position_anno_x = []
    position_anno_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)
        position_anno_x.append((x1 + x0)/2)
        position_anno_y.append((y1 + y0)/2)

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(color='black', width=1),
        
        showlegend=False,
        mode='lines + text', 
        text=weights, 
        textposition='bottom center')
    
    anno_trace = go.Scatter(
        x=position_anno_x, y=position_anno_y,
        
        hoverinfo='none',
        showlegend=False,
        mode='markers + text', 
        text=weights, 
        textposition='bottom center')
    
    # nodes trace
    node_x = []
    node_y = []
    text = []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        text.append(node)
        
    node_trace = go.Scatter(
        x=node_x, y=node_y, text=text,
        mode='markers+text',
        showlegend=False,
        hoverinfo='none',
        marker=dict(
            color='pink',
            size=50,
            line=dict(color='black', width=1)))

    # layout
    layout = dict(plot_bgcolor='white',
                  paper_bgcolor='white',
                  margin=dict(t=10, b=10, l=10, r=10, pad=0),
                  xaxis=dict(linecolor='black',
                             showgrid=False,
                             showticklabels=False,
                             mirror=True),
                  yaxis=dict(linecolor='black',
                             showgrid=False,
                             showticklabels=False,
                             mirror=True))

    # figure
    fig = go.Figure(data=[edge_trace, node_trace, anno_trace], layout=layout)
    
    return fig       



#%%
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

""" Layout specifications for the app
"""
SIDEBAR_STYLE = {
    "position": "fixed",
    "top": 0,
    "left": 0,
    "bottom": 0,
    "width": "20rem",
    "padding": "2rem 1rem",
    "background-color": "#f8f9fa",
}

# the styles for the main content position it to the right of the sidebar and
# add some padding.
CONTENT_STYLE = {
    "margin-left": "18rem",
    "margin-right": "2rem",
    "padding": "2rem 1rem",
}

sidebar = html.Div(
    [
        html.H2("Content", className="display-4"),
        html.Hr(),
 
        dbc.Nav(
            [
                dbc.NavLink("Homepage: Prediction", href="/page-1", active="exact"),
                dbc.NavLink("Page 2: Profile Overview", href="/page-2", active="exact"),
                dbc.NavLink("Page 3: Textual Analysis", href="/page-3", active="exact"),
                dbc.NavLink("Page 4: Image Classification", href="/page-4", active="exact")
            ],
            vertical=True,
            pills=True,
        ),
    ],
    style=SIDEBAR_STYLE,
)


TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#191970'
}


#Tittle + index reference
Content_header1 = dbc.Row(html.Div([
    html.H1('Homepage: Prediction')], style = TEXT_STYLE))
Content_header2 = dbc.Row(html.Div([
    html.H1('Page 2: Profile Overview')], style = TEXT_STYLE))
Content_header3 = dbc.Row(html.Div([
    html.H1('Page 3: Text Analysis')], style = TEXT_STYLE))
Content_header4 = dbc.Row(html.Div([
    html.H1('Page 4: Image Classifications')], style = TEXT_STYLE))


# Page Navigation
index_page = html.Div([
    dcc.Link('Homepage: Prediction', href='/page-1'),
    html.Br(),
    dcc.Link('Profile Overview)', href='/page-2'),
    html.Br(),
    dcc.Link('Textual Analysis', href='/page-3'),
    html.Br(),
    dcc.Link('Image classification', href='/page-4')
    
])

                                                       
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),     
])
"""################################## Page 1: Predicting group ############################
"""
dropdown_network = html.Div([
    dcc.Dropdown( options=sorted(list(all_data_frames.keys())),searchable=True,  multi=True,id='drop_down_network'),
    html.Button('Confirm input',id="Confirm_net", n_clicks=0,  )
    
])


df_sim_matrixx = pd.DataFrame(sim_matrixx, columns = list(picture_dict.keys())[1:] )
df_sim_matrixx.insert(0, 'User', list(picture_dict.keys())[1:])
dash_sim_matrix = dash_table.DataTable(df_sim_matrixx.to_dict('records'), 
                                       [{"name": i, "id": i} for i in df_sim_matrixx.columns], 
                                        fixed_columns={'headers': True, 'data': 1},
                                        
                                        style_data_conditional=[
                                                                {
                                                                    'if': {
                                                                        'filter_query': '{{Person1}} < {}'.format(np.mean(df_sim_matrixx['Person1'])),
                                                                        'column_id': 'Person1'
                                                                    },
                                                                    'backgroundColor': '#FF4136',
                                                                    'color': 'white'
                                                                }, {
                                                                    'if': {
                                                                        'filter_query': '{{Person2}} < {}'.format(np.mean(df_sim_matrixx['Person2'])),
                                                                        'column_id': 'Person2'
                                                                    },
                                                                    'backgroundColor': '#FF4136',
                                                                    'color': 'white'
                                                                }, {
                                                                    'if': {
                                                                        'filter_query': '{{Person3}} < {}'.format(np.mean(df_sim_matrixx['Person3'])),
                                                                        'column_id': 'Person3'
                                                                    },
                                                                    'backgroundColor': '#FF4136',
                                                                    'color': 'white'
                                                                }, {
                                                                    'if': {
                                                                        'filter_query': '{{Person4}} < {}'.format(np.mean(df_sim_matrixx['Person4'])),
                                                                        'column_id': 'Person4'
                                                                    },
                                                                    'backgroundColor': '#FF4136',
                                                                    'color': 'white'
                                                                }, {
                                                                    'if': {
                                                                        'filter_query': '{{Person5}} < {}'.format(np.mean(df_sim_matrixx['Person5'])),
                                                                        'column_id': 'Person5'
                                                                    },
                                                                    'backgroundColor': '#FF4136',
                                                                    'color': 'white'
                                                                }, {
                                                                    'if': {
                                                                        'filter_query': '{{Person6}} < {}'.format(np.mean(df_sim_matrixx['Person6'])),
                                                                        'column_id': 'Person6'
                                                                    },
                                                                    'backgroundColor': '#FF4136',
                                                                    'color': 'white'
                                                                }, {
                                                                    'if': {
                                                                        'filter_query': '{{Person7}} < {}'.format(np.mean(df_sim_matrixx['Person7'])),
                                                                        'column_id': 'Person7'
                                                                    },
                                                                    'backgroundColor': '#FF4136',
                                                                    'color': 'white'
                                                                }, {
                                                                    'if': {
                                                                        'filter_query': '{{Person8}} < {}'.format(np.mean(df_sim_matrixx['Person8'])),
                                                                        'column_id': 'Person8'
                                                                    },
                                                                    'backgroundColor': '#FF4136',
                                                                    'color': 'white'
                                                                }, {
                                                                    'if': {
                                                                        'filter_query': '{{Person9}} < {}'.format(np.mean(df_sim_matrixx['Person9'])),
                                                                        'column_id': 'Person9'
                                                                    },
                                                                    'backgroundColor': '#FF4136',
                                                                    'color': 'white'
                                                                }, {
                                                                    'if': {
                                                                        'filter_query': '{{Person10}} < {}'.format(np.mean(df_sim_matrixx['Person10'])),
                                                                        'column_id': 'Person10'
                                                                    },
                                                                    'backgroundColor': '#FF4136',
                                                                    'color': 'white'
                                                                }, {
                                                                    'if': {
                                                                        'filter_query': '{{Person11}} < {}'.format(np.mean(df_sim_matrixx['Person11'])),
                                                                        'column_id': 'Person11'
                                                                    },
                                                                    'backgroundColor': '#FF4136',
                                                                    'color': 'white'
                                                                }, {
                                                                    'if': {
                                                                        'filter_query': '{{Person12}} < {}'.format(np.mean(df_sim_matrixx['Person12'])),
                                                                        'column_id': 'Person12'
                                                                    },
                                                                    'backgroundColor': '#FF4136',
                                                                    'color': 'white'
                                                                }, {
                                                                    'if': {
                                                                        'filter_query': '{{Person13}} < {}'.format(np.mean(df_sim_matrixx['Person13'])),
                                                                        'column_id': 'Person13'
                                                                    },
                                                                    'backgroundColor': '#FF4136',
                                                                    'color': 'white'
                                                                }, {
                                                                    'if': {
                                                                        'filter_query': '{{Person14}} < {}'.format(np.mean(df_sim_matrixx['Person14'])),
                                                                        'column_id': 'Person14'
                                                                    },
                                                                    'backgroundColor': '#FF4136',
                                                                    'color': 'white'
                                                                }, {
                                                                    'if': {
                                                                        'filter_query': '{{Person15}} < {}'.format(np.mean(df_sim_matrixx['Person15'])),
                                                                        'column_id': 'Person15'
                                                                    },
                                                                    'backgroundColor': '#FF4136',
                                                                    'color': 'white'
                                                                }, {
                                                                    'if': {
                                                                        'filter_query': '{{Person16}} < {}'.format(np.mean(df_sim_matrixx['Person16'])),
                                                                        'column_id': 'Person16'
                                                                    },
                                                                    'backgroundColor': '#FF4136',
                                                                    'color': 'white'
                                                                }, {
                                                                    'if': {
                                                                        'filter_query': '{{Person17}} < {}'.format(np.mean(df_sim_matrixx['Person17'])),
                                                                        'column_id': 'Person17'
                                                                    },
                                                                    'backgroundColor': '#FF4136',
                                                                    'color': 'white'
                                                                }, {
                                                                    'if': {
                                                                        'filter_query': '{{Person18}} < {}'.format(np.mean(df_sim_matrixx['Person18'])),
                                                                        'column_id': 'Person18'
                                                                    },
                                                                    'backgroundColor': '#FF4136',
                                                                    'color': 'white'
                                                                }, {
                                                                    'if': {
                                                                        'filter_query': '{{Person19}} < {}'.format(np.mean(df_sim_matrixx['Person19'])),
                                                                        'column_id': 'Person19'
                                                                    },
                                                                    'backgroundColor': '#FF4136',
                                                                    'color': 'white'
                                                                }, {
                                                                    'if': {
                                                                        'filter_query': '{{Person20}} < {}'.format(np.mean(df_sim_matrixx['Person20'])),
                                                                        'column_id': 'Person20'
                                                                    },
                                                                    'backgroundColor': '#FF4136',
                                                                    'color': 'white'
                                                                }, {
                                                                    'if': {
                                                                        'filter_query': '{{Person21}} < {}'.format(np.mean(df_sim_matrixx['Person21'])),
                                                                        'column_id': 'Person21'
                                                                    },
                                                                    'backgroundColor': '#FF4136',
                                                                    'color': 'white'
                                                                }, {
                                                                    'if': {
                                                                        'filter_query': '{{Person22}} < {}'.format(np.mean(df_sim_matrixx['Person22'])),
                                                                        'column_id': 'Person22'
                                                                    },
                                                                    'backgroundColor': '#FF4136',
                                                                    'color': 'white'
                                                                }, {
                                                                    'if': {
                                                                        'filter_query': '{{Person23}} < {}'.format(np.mean(df_sim_matrixx['Person23'])),
                                                                        'column_id': 'Person23'
                                                                    },
                                                                    'backgroundColor': '#FF4136',
                                                                    'color': 'white'
                                                                }, {
                                                                    'if': {
                                                                        'filter_query': '{{Person24}} < {}'.format(np.mean(df_sim_matrixx['Person24'])),
                                                                        'column_id': 'Person24'
                                                                    },
                                                                    'backgroundColor': '#FF4136',
                                                                    'color': 'white'
                                                                }, {
                                                                    'if': {
                                                                        'filter_query': '{{Person25}} < {}'.format(np.mean(df_sim_matrixx['Person25'])),
                                                                        'column_id': 'Person25'
                                                                    },
                                                                    'backgroundColor': '#FF4136',
                                                                    'color': 'white'
                                                                }, {
                                                                    'if': {
                                                                        'filter_query': '{{Person26}} < {}'.format(np.mean(df_sim_matrixx['Person26'])),
                                                                        'column_id': 'Person26'
                                                                    },
                                                                    'backgroundColor': '#FF4136',
                                                                    'color': 'white'
                                                                }, {
                                                                    'if': {
                                                                        'filter_query': '{{Person27}} < {}'.format(np.mean(df_sim_matrixx['Person27'])),
                                                                        'column_id': 'Person27'
                                                                    },
                                                                    'backgroundColor': '#FF4136',
                                                                    'color': 'white'
                                                                }, {
                                                                    'if': {
                                                                        'filter_query': '{{Person28}} < {}'.format(np.mean(df_sim_matrixx['Person28'])),
                                                                        'column_id': 'Person28'
                                                                    },
                                                                    'backgroundColor': '#FF4136',
                                                                    'color': 'white'
                                                                }, {
                                                                    'if': {
                                                                        'filter_query': '{{Person29}} < {}'.format(np.mean(df_sim_matrixx['Person29'])),
                                                                        'column_id': 'Person29'
                                                                    },
                                                                    'backgroundColor': '#FF4136',
                                                                    'color': 'white'
                                                                }, {
                                                                    'if': {
                                                                        'filter_query': '{{Person30}} < {}'.format(np.mean(df_sim_matrixx['Person30'])),
                                                                        'column_id': 'Person30'
                                                                    },
                                                                    'backgroundColor': '#FF4136',
                                                                    'color': 'white'
                                                                }, {
                                                                    'if': {
                                                                        'filter_query': '{{Person31}} < {}'.format(np.mean(df_sim_matrixx['Person31'])),
                                                                        'column_id': 'Person31'
                                                                    },
                                                                    'backgroundColor': '#FF4136',
                                                                    'color': 'white'
                                                                }, {
                                                                    'if': {
                                                                        'filter_query': '{{Person32}} < {}'.format(np.mean(df_sim_matrixx['Person32'])),
                                                                        'column_id': 'Person32'
                                                                    },
                                                                    'backgroundColor': '#FF4136',
                                                                    'color': 'white'
                                                                }, {
                                                                    'if': {
                                                                        'filter_query': '{{Person33}} < {}'.format(np.mean(df_sim_matrixx['Person33'])),
                                                                        'column_id': 'Person33'
                                                                    },
                                                                    'backgroundColor': '#FF4136',
                                                                    'color': 'white'
                                                                }, {
                                                                    'if': {
                                                                        'filter_query': '{{Person34}} < {}'.format(np.mean(df_sim_matrixx['Person34'])),
                                                                        'column_id': 'Person34'
                                                                    },
                                                                    'backgroundColor': '#FF4136',
                                                                    'color': 'white'
                                                                }, {
                                                                    'if': {
                                                                        'filter_query': '{{Person35}} < {}'.format(np.mean(df_sim_matrixx['Person35'])),
                                                                        'column_id': 'Person35'
                                                                    },
                                                                    'backgroundColor': '#FF4136',
                                                                    'color': 'white'
                                                                }, {
                                                                    'if': {
                                                                        'filter_query': '{{Person36}} < {}'.format(np.mean(df_sim_matrixx['Person36'])),
                                                                        'column_id': 'Person36'
                                                                    },
                                                                    'backgroundColor': '#FF4136',
                                                                    'color': 'white'
                                                                }, {
                                                                    'if': {
                                                                        'filter_query': '{{Person37}} < {}'.format(np.mean(df_sim_matrixx['Person37'])),
                                                                        'column_id': 'Person37'
                                                                    },
                                                                    'backgroundColor': '#FF4136',
                                                                    'color': 'white'
                                                                }, {
                                                                    'if': {
                                                                        'filter_query': '{{Person38}} < {}'.format(np.mean(df_sim_matrixx['Person38'])),
                                                                        'column_id': 'Person38'
                                                                    },
                                                                    'backgroundColor': '#FF4136',
                                                                    'color': 'white'
                                                                }, {
                                                                    'if': {
                                                                        'filter_query': '{{Person39}} < {}'.format(np.mean(df_sim_matrixx['Person39'])),
                                                                        'column_id': 'Person39'
                                                                    },
                                                                    'backgroundColor': '#FF4136',
                                                                    'color': 'white'
                                                                }, {
                                                                    'if': {
                                                                        'filter_query': '{{Person40}} < {}'.format(np.mean(df_sim_matrixx['Person40'])),
                                                                        'column_id': 'Person40'
                                                                    },
                                                                    'backgroundColor': '#FF4136',
                                                                    'color': 'white'
                                                                }, ],
                                            style_table={'minWidth': '100%', 'height': 400, 'overflowY': 'auto'},
                                            fixed_rows={'headers': True},
                                            style_cell = {'minWidth': '90px', 'width': '90px', 'maxWidth': '90px'}
                                       )

alert3 = dbc.Container(dcc.Graph(id='visualisation_block'))
page_1_layout = html.Div([
    html.Div(id='page-1-content'),
    Content_header1,
    html.Br(),
    sidebar,
    html.Br(),
    dbc.Container([dbc.Row([
        dropdown_network]),
        dbc.Row([html.Label("Distance Matrix, red means that the distance is les than the average distance with all other persons for that column"),
                            dash_sim_matrix]),
        dbc.Row([alert3])
        
        
        
        ])
    
                    
]
)
        
"""################################## Page 2: Person comparison ############################
"""
count_list = []
for i in all_data_frames.keys():
    count_list.append(len(all_data_frames[i]['Picture']))

    
df_couts = pd.DataFrame(count_list,  columns = ['Number of Tweets'])
fig_counts = px.histogram(df_couts, x='Number of Tweets')

dropdown1 = html.Div([
    dcc.Dropdown(options=sorted(list(all_data_frames.keys())), id='demo-dropdown1'),
    html.Div(id='dd-output-container1')
])
table1 = html.Div(id='table1-container')
alert1 = dbc.Container(id='tbl_out1')

dropdown2 = html.Div([
    dcc.Dropdown(options=sorted(list(all_data_frames.keys())), id='demo-dropdown2'),
    html.Div(id='dd-output-container2')
])
table2 = html.Div(id='table2-container')
alert2 = dbc.Container(id='tbl_out2')
alert5 = dbc.Container(dcc.Graph(figure=fig_counts))


page_2_layout = html.Div([
    html.Div(id='page-2-content'),
    Content_header2,
    html.Br(),
    sidebar,
    dbc.Container([
        
        dbc.Row([
            
            alert5
            ]),
        dbc.Row([
        # Boxplot
        dbc.Col([dropdown1,
                 table1,
                 alert1

                 ]),

        dbc.Col([dropdown2,
                 table2,
                 alert2

                 ]),
    ]),

    ]

    )

])

"""################################## Page 3: Textual Analysis ############################
"""
page_3_layout = html.Div([
    html.Div(id='page-3-content'),
    Content_header3,
    html.Br(),
    sidebar

])

"""################################## Page 4: Picture Analysis ############################
"""

count_list = []
for i in all_data_frames.keys():
    count_list.append(len(all_data_frames[i]['Picture']))

    
df_couts = pd.DataFrame(count_list,  columns = ['Number of Tweets'])
fig_counts = px.histogram(df_couts, x='Number of Tweets')

dropdown41 = html.Div([
    dcc.Dropdown(options=sorted(list(all_data_frames.keys())), id='demo-dropdown41'),
    html.Div(id='dd-output-container41')
])
table41 = html.Div(id='table41-container')
alert41 = dbc.Container(id='tbl_out41')
alert3333 = dbc.Container(id='hist_thingy')
page_4_layout = html.Div([
    html.Div(id='page-4-content'),
    Content_header4,
    html.Br(),
    sidebar,
    dbc.Container([dbc.Row([
        # Boxplot
        dbc.Col([dropdown41,
                 table41

                 ]),

        dbc.Col([alert3333,
            alert41

                 ]),
    ]),

    ]

    )

])


"""################################## Callbacks Home-page: Navigation ############################
"""
@app.callback(dash.dependencies.Output("page-content", "children"), [dash.dependencies.Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/page-1":
        return page_1_layout
    elif pathname == "/":
        return page_1_layout
    elif pathname == "/page-2":
        return page_2_layout
    elif pathname == "/page-3":
        return page_3_layout
    elif pathname == "/page-4":
        return page_4_layout
    # If the user tries to reach a different page, return a 404 message
    return html.Div(
        [
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {pathname} was not recognised..."),
        ],
        className="p-3 bg-light rounded-3",
    )


"""################################## Callbacks page11: Table and network visualization ############################
"""

@app.callback(
    Output("visualisation_block", "figure"),
    Input("Confirm_net", "n_clicks"),
    Input("drop_down_network", "value"))
def update_vis(n_clicks, value):
    if n_clicks:
        edge_list = []
        allowed = []
        for i in value:
            allowed.append(int(i[6:]))
            
        # allowed = [1,2,3,8,10, 40, 41, 33]
        for i in range(len(sim_matrixx)):
            if i+1 in allowed:
                
                user1 = f'Person{i+1}'
                for j in range(len(sim_matrixx[i])):
                    if j +1 in allowed:
                        
                        user2 = f'Person{j+1}'
                        edge_list.append((user1, user2,sim_matrixx[i][j]))

        fig = plot_network_graph(edge_list)
        return fig
    else:
        raise PreventUpdate


"""################################## Callbacks page11: Person X compared to Person Y ############################
"""

"################################### User y ###################################"
@callback(Output('tbl_out1', 'children'),
          [Input('table1', 'active_cell'), Input('table1', 'data')])
def update_graphs(value, data):
    if value and data:
        row = value['row']
        col_id = value['column_id']
        iets = data[row][col_id]

        caption = iets[:-4] + 'caption.txt'
        if caption in all_captions.keys():
            text = all_captions[caption]
        else:
            text = 'No caption was supplied'
        fig = px.imshow(np.array(all_pictures[iets]), title="Caption:" + text)
    else:
        raise PreventUpdate

    return dcc.Graph(figure=fig) if value else "Click the table"


@app.callback(
    Output('dd-output-container1', 'children'), Output('table1-container', 'children'),
    Input('demo-dropdown1', 'value')
)
def update_output(value):
    if value:
        df = all_data_frames[value]
        q = dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns], id='table1', 
                                 
                                     style_table={'minWidth': '100%', 'height': 400, 'overflowY': 'auto'},
                                     fixed_rows={'headers': True},
                                     style_cell = {'minWidth': '90px', 'width': '90px', 'maxWidth': '90px'}
                                 )
        return f'You have selected {value}', q
    else:
        raise PreventUpdate

    return f'You have selected {value}', q


"################################### User x ###################################"
@callback(Output('tbl_out2', 'children'),
          [Input('table2', 'active_cell'), Input('table2', 'data')])
def update_graphs2(value, data):
    if value and data:
        row = value['row']
        col_id = value['column_id']
        iets = data[row][col_id]

        caption = iets[:-4] + 'caption.txt'
        if caption in all_captions.keys():
            text = all_captions[caption]
        else:
            text = 'No caption was supplied'
        fig = px.imshow(np.array(all_pictures[iets]), title="Caption:" + text)
    else:
        raise PreventUpdate

    return dcc.Graph(figure=fig) if value else "Click the table"


@app.callback(
    Output('dd-output-container2', 'children'), Output('table2-container', 'children'),
    Input('demo-dropdown2', 'value')
)
def update_output2(value):
    if value:
        df = all_data_frames[value]
        q = dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns], id='table2',                                  
                                     style_table={'minWidth': '100%', 'height': 400, 'overflowY': 'auto'},
                                     fixed_rows={'headers': True},
                                     style_cell = {'minWidth': '90px', 'width': '90px', 'maxWidth': '90px'})
        return f'You have selected {value}', q
    else:
        raise PreventUpdate

    return f'You have selected {value}', q


"""################################### Callbacks page 3: ##################################"""


"""################################### Callbacks page 4: Showing classification next to picture#################"""
@callback(Output('tbl_out41', 'children'),
          [Input('table41', 'active_cell'), Input('table41', 'data')])
def update_graphs4(value, data):
    if value and data:
        row = value['row']
        col_id = value['column_id']
        iets = str(data[row][col_id])


        top_5_predictions = predictions_df[predictions_df['image_file'] == iets].sort_values(by='rank')
        top_5_list = [
            (row['class_label'], f"{row['probability'] * 100:.2f}%") for _, row in top_5_predictions.iterrows()
        ]

        caption = iets[:-4] + 'caption.txt'
        if caption in all_captions.keys():
            text = all_captions[caption]
        else:
            text = 'No caption was supplied'
        fig = px.imshow(np.array(all_pictures[iets]), title=f"Caption: {text}")
        fig.update_layout(
            title=dict(
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top',
                font=dict(size=14),
            ),
        )
        prediction_table = html.Table([
            html.Tr([html.Th('Top 5 Predictions')]),
            *[html.Tr([html.Td(label), html.Td(prob)]) for label, prob in top_5_list]
        ])
        return [dcc.Graph(figure=fig), prediction_table]
    else:
        raise PreventUpdate

    return  "Click the table"


@app.callback(
    Output('dd-output-container41', 'children'), Output('table41-container', 'children'),
    Input('demo-dropdown41', 'value')
)
def update_output4(value):
    if value:
        df = all_data_frames[value]
        q = dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns], id='table41')
        return f'You have selected {value}', q
    else:
        raise PreventUpdate

    return f'You have selected {value}', q

@callback(Output('hist_thingy', 'children'),
          [Input('demo-dropdown41', 'value')])

def update_output8(value):
    if value:
        classifications_df_pers =predictions_df[predictions_df['person_id'] == int(value[6:])]
        
        fig_counts_hist_pers = px.histogram(classifications_df_pers, x='class_label')
        return dcc.Graph(figure=fig_counts_hist_pers)
    else:
        raise PreventUpdate

    return f'You have selected {value}'

"""Running dashboard"""

if __name__ == '__main__':
    app.run_server(debug=True, )

    
