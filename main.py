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

import dash                     #(version 1.0.0)
from dash import dcc
from dash import html, dash_table
import dash_bootstrap_components as dbc
import os



import pandas as pd
from dash import Input, Output, dcc, html
from dash import Dash, Input, Output, callback, dash_table
import pandas as pd
import dash_bootstrap_components as dbc

import dash
import dash_core_components as dcc
from dash import html
from dash.dependencies import Input, Output
import os 
from PIL import Image
import numpy as np
import plotly.express as px
from dash.exceptions import PreventUpdate
import matplotlib.pyplot as plt
import mpld3
#%%

""" Importing Data """
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
######## Making Dataframe per person
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
            




#%%
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)

# the style arguments for the sidebar. We use position:fixed and a fixed width
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
################################## Page 1 ############################

page_1_layout = html.Div([
    html.Div(id='page-1-content'),
    Content_header1,
    html.Br(),
    sidebar,
    html.Br()
                    
]
)

##################################  Page 2    ################################## 
df = all_data_frames['Person7']
table  = dash_table.DataTable(df.to_dict('records'), [{"name": i, "id": i} for i in df.columns], id = 'overview') 
alert = dbc.Container(id='tbl_out')

page_2_layout = html.Div([
    html.Div(id='page-2-content'),
    Content_header2,
    html.Br(),
    sidebar,
    dbc.Container([table
        
        ]),
    dbc.Container([alert,
                   
                   ])

])




##################################  Page 3   ################################## 
page_3_layout = html.Div([
    html.Div(id='page-3-content'),
    Content_header3,
    html.Br(),
    sidebar

])            
       
##################################  Page 4   ################################## 

page_4_layout = html.Div([
    html.Div(id='page-4-content'),
    Content_header4,
    html.Br(),
    sidebar
    
])         
                     
################################### Callbacks Home-page ##################################
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
    
                     
################################### Callbacks page 1 ##################################


################################### Callbacks page 2 ##################################
import base64
@callback(Output('tbl_out', 'children'), 
          [Input('overview', 'active_cell'),Input('overview', 'data')])
def update_graphs(value,data):
    
    row = value['row']
    column = value['row']
    col_id = value['column_id']
    iets = data[row][col_id]
    

    
    caption = iets[:-4] + 'caption.txt'
    if caption in all_captions.keys():
        text = all_captions[caption]
    else:
        text = 'No caption was supplied'
    fig = px.imshow(np.array(all_pictures[iets]), title  = "Caption:" + text)
    
    return  dcc.Graph(figure=fig) if value else "Click the table"



################################### Callbacks page 3 ##################################
# html callback function to hover the data on specific coordinates
# @app.callback(
#    Output('image', 'src'),
#    Input('overview2', 'active_cell'))
# def open_url(hoverData):
#    if hoverData:
#       return hoverData["points"][0]["customdata"][0]
#    else:
#       raise PreventUpdate
################################### Callbacks page 4 ##################################



############## Runnen van dashboard #####################

    
if __name__ == '__main__':

    app.run_server(debug=True,)
    

    
