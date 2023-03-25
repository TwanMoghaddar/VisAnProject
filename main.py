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
import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, dcc, html
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
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

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
        html.H2("Conent", className="display-4"),
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
    html.Br(),
    # Side_bar_layout1,
    # Graph_layout,
    
                    
]
)

##################################  Page 2    ################################## 
page_2_layout = html.Div([
    html.Div(id='page-2-content'),
    Content_header2,
    html.Br(),
    sidebar,
    # Side_bar_layout2,
    # stan_graphhhh
])
##################################  Page 3   ################################## 
page_3_layout = html.Div([
    html.Div(id='page-3-content'),
    Content_header3,
    html.Br(),
    sidebar,

])            
       
##################################  Page 4   ################################## 

page_4_layout = html.Div([
    html.Div(id='page-4-content'),
    Content_header4,
    html.Br(),
    sidebar
    
])         
                     
################################### Callbacks Home-page ##################################
@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def render_page_content(pathname):
    if pathname == "/page-1":
        return html.P(page_1_layout)
    elif pathname == "/page-2":
        return html.P(page_2_layout)
    elif pathname == "/page-3":
        return html.P(page_3_layout)
    elif pathname == "/page-4":
        return html.P(page_4_layout)
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


################################### Callbacks page 3 ##################################

################################### Callbacks page 4 ##################################



############## Runnen van dashboard #####################

if __name__ == '__main__':
    app.run_server(debug=False)
    
    
