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
from dash import html
import dash_bootstrap_components as dbc
import os
#%%
""" Importing Data """
path_to_all_data = r"C:\Users\caspe\OneDrive - TU Eindhoven\DS&AI 2022-2023\Q3\2AMV10 - Visual Analytics\2. Terrorists at CGCS\data\data"


from PIL import Image
#importing the Image class from PIL package
def picture_import(path):
    person_images = {}
    person_text = {}
    for subdir, dirs, files in os.walk(path_to_all_data):
        person = subdir[116:] #This is different for everyone as folders are named differently!!
        person_images[person] = []
        person_text[person] = []

        for file in files:
            im_path = os.path.join(subdir, file)
            str_im_path = str(im_path)
            if str_im_path[-1] == 'g':
                
                #read the image, creating an object
                im = Image.open(im_path)
                person_images[person].append(im)
            if str_im_path[-1] == 't':
                with open(str_im_path, 'r') as file:
                    data = file.read().replace('\n', '')
                    person_text[person].append(data)
    return person_images, person_text

data_import = picture_import(path_to_all_data)
picture_dict = data_import[0]
text_dict = data_import[1]



#%%
app = dash.Dash(__name__, suppress_callback_exceptions=True)

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content'),     
])


CONTENT_STYLE = {
    'margin-left': '25%',
    'margin-right': '5%',
    'top': 0,
    'padding': '20px 10px'
}

# # Side bar page 1
TEXT_STYLE = {
    'textAlign': 'center',
    'color': '#191970'
}

SIDEBAR_STYLE = {
    'position': 'fixed',
    'top': 0,
    'left': 0,
    'bottom': 0,
    'width': '20%',
    'padding': '20px 10px',
    'background-color': '#f8f9fa'
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

                                                       
    
################################## Page 1 ############################

page_1_layout = html.Div([
    html.Div(id='page-1-content'),
    Content_header1,
    html.Br(),
    index_page,
    html.Br(),
    # Side_bar_layout1,
    # Graph_layout,
    
                    
], style=CONTENT_STYLE
)

##################################  Page 2    ################################## 
page_2_layout = html.Div([
    html.Div(id='page-2-content'),
    Content_header2,
    html.Br(),
    index_page,
    # Side_bar_layout2,
    # stan_graphhhh
])
##################################  Page 3   ################################## 
page_3_layout = html.Div([
    html.Div(id='page-3-content'),
    Content_header3,
    html.Br(),
    index_page,

])            
       
##################################  Page 4   ################################## 

page_4_layout = html.Div([
    html.Div(id='page-4-content'),
    Content_header4,
    html.Br(),
    index_page,
    
])         
                     
################################### Callbacks Home-page ##################################
@app.callback(dash.dependencies.Output('page-content', 'children'),
              [dash.dependencies.Input('url', 'pathname')])

def display_page(pathname):
    if pathname == '/page-1':
        return page_1_layout
    elif pathname == '/page-2':
        return page_2_layout
    elif pathname == '/page-3':
        return page_3_layout
    
    elif pathname == '/page-4':
        return page_4_layout
    else:
        return page_1_layout
    
                     
################################### Callbacks page 1 ##################################

        
################################### Callbacks page 2 ##################################


################################### Callbacks page 3 ##################################

################################### Callbacks page 4 ##################################



############## Runnen van dashboard #####################

if __name__ == '__main__':
    app.run_server(debug=False)
    
    
