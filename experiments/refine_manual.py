import sys
import os
import argparse
import numpy as np
import librosa
import json

import dash
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
import dash_html_components as html
import dash_core_components as dcc
import dash_audio_components
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from dcase_models.util.gui import encode_audio
from dcase_models.data.features import MelSpectrogram
from dcase_models.data.datasets import UrbanSound8k

from dcase_models.util.files import load_json, mkdir_if_not_exists, save_pickle, load_pickle
from dcase_models.util.data import evaluation_setup
from dcase_models.data.data_generator import DataGenerator

sys.path.append('../')
from apnet.gui import generate_figure2D 
from apnet.gui import generate_figure_weights
from apnet.gui import generate_figure_mel
from apnet.model import APNet
from apnet.layers import PrototypeLayer, WeightedSum
from apnet.datasets import MedleySolosDb, GoogleSpeechCommands

from tensorflow import get_default_graph

available_models = {
    'APNet' :  APNet,
}

available_features = {
    'MelSpectrogram' :  MelSpectrogram,
}

available_datasets = {
    'UrbanSound8k' :  UrbanSound8k,
    'MedleySolosDb' : MedleySolosDb,
    'GoogleSpeechCommands' : GoogleSpeechCommands
}

# Define app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SPACELAB])

graph = get_default_graph()

# Generate layout

parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter
)
parser.add_argument(
    '-d', '--dataset', type=str,
    help='dataset name (e.g. UrbanSound8k, ESC50, URBAN_SED, SONYC_UST)',
    default='UrbanSound8k'
)
parser.add_argument(
    '-f', '--features', type=str,
    help='features name (e.g. Spectrogram, MelSpectrogram, Openl3)',
    default='MelSpectrogram'
)
parser.add_argument(
    '-mp', '--models_path', type=str,
    help='path to load the trained model',
    default='./'
)
parser.add_argument(
    '-dp', '--dataset_path', type=str,
    help='path to load the dataset',
    default='./'
)
parser.add_argument(
    '-m', '--model', type=str,
    help='model name (e.g. MLP, SB_CNN, SB_CNN_SED, A_CRNN, VGGish)')

parser.add_argument('-fold', '--fold_name', type=str, help='fold name',
                    default='fold1')

parser.add_argument('--o', dest='overwrite', action='store_true')
parser.set_defaults(overwrite=False)

parser.add_argument('-gpu', '--gpu_visible', type=str, help='gpu_visible',
                    default='0')

parser.add_argument('--force', dest='force_prototpye_calculation', action='store_true')
parser.set_defaults(force_prototpye_calculation=False)

parser.add_argument('--wo_audio', dest='get_audio_prototypes', action='store_false')
parser.set_defaults(get_audio_prototypes=True)

args = parser.parse_args()

# only use one GPU
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_visible

dataset_name = args.dataset
model_name = args.model
features_name = args.features

# Model paths
model_input_folder = os.path.join(args.models_path, args.dataset, args.model)

# Get parameters
parameters_file = os.path.join(model_input_folder, 'config.json')
params = load_json(parameters_file)

params_dataset = params['datasets'][dataset_name]
params_features = params['features']


model_containers = {}
fold_name = args.fold_name

exp_folder_fold = os.path.join(model_input_folder, fold_name)

if args.overwrite:
    exp_folder_output = model_input_folder
else:
    exp_folder_output = os.path.join(model_input_folder, 'refine_manual')

exp_folder_output_fold = os.path.join(exp_folder_output, fold_name)
mkdir_if_not_exists(exp_folder_output_fold, parents=True)
print(args.overwrite, exp_folder_output_fold)

dataset_class = available_datasets[dataset_name]
dataset_path = os.path.join(args.dataset_path, params_dataset['dataset_path'])
dataset = dataset_class(dataset_path)

# Get and init feature class
features_class = available_features[features_name]
features = features_class(**params_features[features_name])

features.extract(dataset)

kwargs = {'custom_objects': {'PrototypeLayer': PrototypeLayer, 'WeightedSum': WeightedSum}}
with graph.as_default():
    model_containers[fold_name] = APNet(
        model=None, model_path=exp_folder_fold, metrics=['classification'],
        **kwargs, **params['models']['APNet']['model_arguments']
    )
    model_containers[fold_name].load_model_weights(exp_folder_fold)

scaler_path = os.path.join(exp_folder_fold, 'scaler.pickle') 
scaler = load_pickle(scaler_path)

scaler_path = os.path.join(exp_folder_output_fold, 'scaler.pickle') 
save_pickle(scaler, scaler_path)

folds_train, folds_val, _ = evaluation_setup(
    fold_name, dataset.fold_list,
    params_dataset['evaluation_mode'],
    use_validate_set=True
)

data_gen_train = DataGenerator(
    dataset, features, folds=folds_train,
    batch_size=32,
    shuffle=True, train=True, scaler=scaler
)

if args.dataset == 'MedleySolosDb':
    data_gen_train.audio_file_list = data_gen_train.audio_file_list[:int(len( data_gen_train.audio_file_list)/3)]
if args.dataset ==  'GoogleSpeechCommands':
    data_gen_train.audio_file_list = data_gen_train.audio_file_list[:int(len( data_gen_train.audio_file_list)/10)]
#for j in range(len(data_gen_train)):
X_train, Y_train = data_gen_train.get_data()

file_list = []
for file_dict in data_gen_train.audio_file_list:
    file_list.append(file_dict['file_original'])

# Take first sequence of each file
#for j in range(len(X_train)):
#    X_train[j] = X_train[j][0]
#    Y_train[j] = Y_train[j][0]

#X_train = np.asarray(X_train)
#Y_train = np.asarray(Y_train)

model_input_to_embeddings = model_containers[fold_name].model_input_to_embeddings()

X_feat = model_input_to_embeddings.predict(X_train)[0]

print(X_train.shape, Y_train.shape, X_feat.shape, len(file_list))

data_instances_path = os.path.join(exp_folder_fold, 'data_instances.pickle')
prototypes_path = os.path.join(exp_folder_fold, 'prototypes.pickle')

projection2D = None
if os.path.exists(prototypes_path):
    model_containers[fold_name].prototypes = load_pickle(prototypes_path)
    projection2D = model_containers[fold_name].prototypes.projection2D

model_containers[fold_name].get_data_instances(X_feat, X_train, Y_train, file_list, projection2D=projection2D)

    # model_containers[fold_name].data_instances = load_pickle(data_instances_path)

if (not os.path.exists(prototypes_path)) or (args.force_prototpye_calculation):
    convert_audio_params = None
    if args.get_audio_prototypes:
        convert_audio_params = {
            'sr': params_features[args.features]['sr'],
            'scaler': scaler,
            'mel_basis': features.mel_basis,
            'audio_hop': params_features[args.features]['audio_hop'],
            'audio_win': params_features[args.features]['audio_win']
        }

    model_containers[fold_name].get_prototypes(
        X_train,
        projection2D=model_containers[fold_name].data_instances.projection2D,
        convert_audio_params=convert_audio_params
    )
    save_pickle(model_containers[fold_name].prototypes, prototypes_path)
    # save_pickle(model_containers[fold_name].data_instances, data_instances_path)

label_list = dataset.label_list.copy()
for j in range(len(label_list)):
    label_list[j] = label_list[j].replace('_', ' ')

print(label_list)

figure2D = generate_figure2D(
    model_containers[fold_name], x_select=0, y_select=1,
    samples_per_class=10, label_list=label_list
)

fig_weights = generate_figure_weights(model_containers[fold_name], label_list=label_list)

_,center_mel_blank,_,_,_,_  = model_containers[fold_name].data_instances.get_instance_by_index(0)
fig_mel = generate_figure_mel(center_mel_blank)

options = []
for j in range(4):
    options.append({'label':' PCA Component '+str(j+1),'value':j})
x_select = dcc.Dropdown(id='x_select',options=options,value=0,style={'width': '100%'})
y_select = dcc.Dropdown(id='y_select',options=options,value=1,style={'width': '100%'})

slider_samples = html.Div(
    dcc.Slider(id='samples_per_class',min=1, max=500, step=1, value=10, vertical=False),
    style={'width':'100%'}
)

plot2D = dcc.Graph(
    id='plot2D', figure=figure2D,
    style={"height" : "100%", "width" : "100%"}
)
plot_mel = dcc.Graph(
    id="plot_mel", figure=fig_mel,
    style={"width": "100%", "display": "inline-block",'float':'left'}
)
plot_weights = dcc.Graph(
    id="plot_weights", figure = fig_weights, 
    style={"width": "100%", "display": "inline-block"}
)
audio_player = dash_audio_components.DashAudioComponents(
    id='audio-player', src="", autoPlay=False, controls=True, style={'width':'70%'})

button_save = dbc.Button(
    'Save model',
    id='save_model',
    n_clicks_timestamp=0,
    style={'width':'70%'},
    color="primary",
    className="mr-1"
)

msg_model = dbc.Alert(
    "Messages about model",
    id="msg_model",
    is_open=False,
    duration=4000,
)


button_delete = dbc.Button(
    'Delete prototype',
    id='delete_and_convert',
    n_clicks_timestamp=0,
    style={'display':'none','width':'70%'},
    color="primary",
    className="mr-1"
)

tab_visualization = html.Div([
    msg_model,
    dbc.Row(
        [
            dbc.Col(html.Div([plot2D]), width=8),
            dbc.Col(
                [
                    dbc.Row([plot_mel], align='center'),
                    dbc.Row([audio_player], align='center'),html.Br(),
                    dbc.Row([button_save], align='center'),
                    dbc.Row([button_delete], align='center'),html.Br(),
                    #dbc.Row([x_select, y_select], align='center'),html.Br(),
                    #dbc.Row([], align='center'),html.Br(),
                    #dbc.Row([slider_samples], align='center')
                ], width=2),
        ]
    ),
   dbc.Row(
       [
 #          dbc.Col(html.Div([]), width=1),
           dbc.Col(html.Div([x_select]), width=2),
           dbc.Col(html.Div([y_select]), width=2),
           dbc.Col(html.Div([slider_samples]), width=3)
       ]
   ),
    dbc.Row(
        [
            dbc.Col(html.Div([plot_weights]), width=8)
        ]
    )
])

tab_train = html.Div([])
tab_evaluation = html.Div([])

# Define Tabs
tabs = dbc.Tabs(
    [
        #dbc.Tab(tab_model, label="Model definition", tab_id='tab_model'),
        dbc.Tab(tab_visualization, label="Data visualization",
                tab_id='tab_visualization'),
        dbc.Tab(tab_train, label="Model training", tab_id='tab_train'),
        dbc.Tab(tab_evaluation, label="Model evaluation",
                tab_id='tab_evaluation'),
        #dbc.Tab(tab_demo, label="Prediction visualization",
        #        tab_id='tab_demo'),
    ], id='tabs'
)

# Define layout
app.layout = html.Div([tabs])

## Callbacks

@app.callback(
    Output('plot2D', 'figure'),
    [Input('x_select', 'value'),
    Input('y_select', 'value'),
    Input('samples_per_class', 'value'),
    Input('plot_weights', 'clickData'),
    Input('delete_and_convert', 'n_clicks_timestamp')],
    [State('plot2D', 'selectedData')]
    )
def reload_figure2D(x_select, y_select, samples_per_class, clickData,
                    n_clicks_timestamp, selectedData):
    selectedpoints = []
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if (button_id == 'plot_weights') and isinstance(clickData,dict):
        selectedpoints = [clickData['points'][0]['x']]
    if ((button_id == 'delete_and_convert') and (n_clicks_timestamp > 0)
        and (isinstance(selectedData,dict))):
        point = np.array(
            [selectedData['points'][0]['x'],
            selectedData['points'][0]['y']]
            )
        dist_protos = model_containers[fold_name].prototypes.get_distances(
            point, components=(x_select, y_select))
        dist_data = model_containers[fold_name].data_instances.get_distances(
            point, components=(x_select, y_select))
        if np.amin(dist_data) <= np.amin(dist_protos): # click on k-mean
            arg_dist = np.argmin(dist_data)    
            (center_feat,center_mel,center_2D,
             center_class,_,center_file) = model_containers[fold_name].data_instances.remove_instance(arg_dist)
            data, sr = librosa.core.load(center_file)  
            center_audio = {'data':data, 'sr': sr}          
            model_containers[fold_name].prototypes.add_instance(
                int(center_class), center_mel,center_feat,
                embedding2D=center_2D, audio=center_audio)  
        else:
            arg_dist = np.argmin(dist_protos)
            model_containers[fold_name].prototypes.remove_instance(arg_dist)         
        
        with graph.as_default():
            model_containers[fold_name].update_model_to_prototypes() 

    figure2D = generate_figure2D(
        model_containers[fold_name],
        x_select=x_select, y_select=y_select,
        samples_per_class=samples_per_class,
        label_list=label_list,
        selectedpoints=selectedpoints
    )
    return figure2D


@app.callback(
    Output('plot_weights', 'figure'),
    [Input('delete_and_convert', 'n_clicks_timestamp')],
    )
def reload_figure_weights(n_clicks_timestamp):
    fig_weights = generate_figure_weights(model_containers[fold_name], label_list=label_list)
    return fig_weights

@app.callback(
    [Output('plot_mel', 'figure'),
     Output('audio-player', 'overrideProps'),
     Output('delete_and_convert', 'children'),
     Output('delete_and_convert', 'style')],
    [Input('plot2D', 'selectedData'),
    Input('plot_weights', 'clickData')],
    [State('x_select', 'value'),
    State('y_select', 'value')])
def click_on_plot2d(clickData, clickData_weights, x_select, y_select):
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if (button_id == 'plot_weights') and isinstance(clickData_weights,dict):
        _,_,proto2D,_,_ = model_containers[fold_name].prototypes.get_instance_by_index(
            clickData_weights['points'][0]['x'])
        point = proto2D[[x_select, y_select]]
    elif (button_id == 'plot2D') and isinstance(clickData,dict):
        point = np.array([
            clickData['points'][0]['x'], clickData['points'][0]['y']
        ])
    else:
        return [
            generate_figure_mel(center_mel_blank),
            {'autoPlay': False, 'src': ''},
             "Select a point",
             {'display':'none','width':'70%'}
        ]

    dist_protos = model_containers[fold_name].prototypes.get_distances(
        point,components=(x_select, y_select))
    dist_data = model_containers[fold_name].data_instances.get_distances(
        point,components=(x_select, y_select))
    if np.amin(dist_data) <= np.amin(dist_protos): # click on k-mean
        arg_dist = np.argmin(dist_data)  
        (center_feat,center_mel,center_2D,
        center_class,center_audio,center_file) = model_containers[fold_name].data_instances.get_instance_by_index(arg_dist)  
        
        figure = generate_figure_mel(center_mel)
        
        data, sr = librosa.core.load(center_file)
        return [
            figure, 
            {'autoPlay': True, 'src': encode_audio(data, sr)},
            "Convert center to Prototype",
            {'display':'inline-block','width':'70%'}
        ]
    else:
        arg_dist = np.argmin(dist_protos)
        (proto_feat, proto_mel,
        proto_2D, proto_class, proto_audio) = model_containers[fold_name].prototypes.get_instance_by_index(arg_dist)

        figure = generate_figure_mel(proto_mel)

        if args.get_audio_prototypes:
            audio_output = {'autoPlay': True, 'src': encode_audio(proto_audio['data'], proto_audio['sr'])},
        else:
            audio_output = {'autoPlay': False, 'src': ''},
                
        return [
            figure, 
            audio_output,
            "Delete Prototype",
            {'display':'inline-block','width':'70%'}
        ]

@app.callback(
    [Output("msg_model", "is_open"),
     Output("msg_model", "children"),
     Output("msg_model", "color")],
    [Input('save_model', 'n_clicks_timestamp')],
    )
def save_model(n_clicks_timestamp):
    if n_clicks_timestamp > 0:
        print(exp_folder_output_fold)
        prototypes_path = os.path.join(exp_folder_output_fold, 'prototypes.pickle')
        model_containers[fold_name].save_model_json(exp_folder_output_fold)
        model_containers[fold_name].save_model_weights(exp_folder_output_fold)

        # Save new params
        params_path = os.path.join(exp_folder_output, 'config.json') 

        params['models']['APNet']['train_arguments']['init_last_layer'] = 0
        params['models']['APNet']['model_arguments']['n_prototypes'] = -1

        with open(params_path, 'a') as outfile:
            json.dump(params, outfile, indent=2)

        return [True, 'Model saved', 'success']
    else:
        return [False, '', 'success']

if __name__ == '__main__':
    app.run_server(debug=False)