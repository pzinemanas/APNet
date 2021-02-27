import numpy as np
       
from io import BytesIO
import wave
import struct

from dcase_models.util.gui import encode_audio


#from .utils import save_model_weights,save_model_json, get_data_train, get_data_test
#from .utils import init_model, evaluate_model, load_scaler, save, load
#from .model import debugg_model, prototype_loss,prototypeCNN_maxpool
#from .prototypes import Prototypes

import os
from keras.callbacks import ModelCheckpoint,CSVLogger
from keras.optimizers import Adam
import keras.backend as K

import dash
from dash.dependencies import Input, Output
import dash_html_components as html
import dash_core_components as dcc
import dash_audio_components
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import librosa
import sys
from dcase_models.util.files import save_pickle, load_pickle
from dcase_models.util.data import get_fold_val
#from dcase_models.model.model import debugg_model, modelAPNet

colors = ['#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728', 
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf']

class_names = (['air conditioner', 'car horn', 'children playing',
               'dog bark', 'drilling', 'engine idling', 'gun shot',
               'jack- hammer', 'siren', 'street music'])  
class_names2 = (['air<br>conditioner', 'car<br>horn', 'children<br>playing',
               'dog<br>bark', 'drilling', 'engine<br>idling', 'gun<br>shot',
               'jack-<br>hammer', 'siren', 'street<br>music'])  
class_names_av = ['AC', 'CH', 'CP', 'DB', 'DR', 'EI', 'GS', 'JA', 'SI', 'SM']

#from audio_prototypes.utils import load_training_log
from shutil import copyfile

import matplotlib.pyplot as plt
cm = plt.get_cmap('viridis')



def generate_figure2D(model_container, selectedpoints=[],
                      x_select=0, y_select=1, samples_per_class=10,
                      label_list=[]):
    prototypes_feat,prototypes_mel,protoypes2D,prototypes_classes,_ = model_container.prototypes.get_all_instances()

    n_classes = len(label_list) if len(label_list) > 0 else 10

    prototype_ixs = np.arange(0,len(prototypes_feat))
    x = []
    y = []
    classes = []
    classes_ix = []
    prototypes_ixs = []
    for class_ix in range(n_classes):
        prototypes_class_ix = protoypes2D[prototypes_classes == class_ix]
        prototype_ixs_class = prototype_ixs[prototypes_classes == class_ix]
        #print(prototypes_class_ix.shape)
        xj = []
        yj = []
        classesj = []
        for j in range(len(prototypes_class_ix)):
            xj.append(prototypes_class_ix[j, x_select])
            yj.append(prototypes_class_ix[j, y_select])
            classesj.append('prototype'+str(prototype_ixs_class[j]))
            # classes_ix.append(int(prototypes_classes[j]))
        x.append(xj)
        y.append(yj)
        classes.append(classesj)    
        prototypes_ixs.append(prototype_ixs_class)


    centers_feat,centers_mel,centers2D,centers_classes,centers_audio,centers_file_names = model_container.data_instances.get_all_instances()
    centers_ixs = np.arange(0,len(centers2D))
    x_centers = []
    y_centers = []
    classes_centers = []
    classes_ix_centers = []

    ### Add this to tests. Delete!!!
    #centers2D = self.model_containers[self.fold_test].data_instances.X_feat_2D['X']#self.X_2D[self.fold_test]['X']
    #centers_classes = self.model_containers[self.fold_test].data_instances.X_feat_2D['Y']
    #centers_ixs = np.arange(0,len(centers2D))

    for class_ix in range(n_classes):
        centers_class_ix = centers2D[centers_classes == class_ix]
        centers_ixs_class = centers_ixs[centers_classes == class_ix]
        xj = []
        yj = []
        classesj = []
        for j in range(len(centers_class_ix)):
            xj.append(centers_class_ix[j, x_select])
            yj.append(centers_class_ix[j, y_select])
            classesj.append('center'+str(centers_ixs_class[j]))
            # classes_ix.append(int(prototypes_classes[j]))
        x_centers.append(xj)
        y_centers.append(yj)
        classes_centers.append(classesj)  



    fig = make_subplots(rows=1, cols=1)#, column_widths=[0.8, 0.2])
    size = 10

    proto_list = []
    for label in label_list:
        proto_list.append(label + ' (protos.)')

    for j in range(n_classes):
        s = min(samples_per_class,len(x_centers[j]))
        selectedpoints_j = None
        if len(selectedpoints) > 0:
            proto_ixs = prototypes_ixs[j]
            selectedpoints_j = []
            for point in selectedpoints:
                if point in proto_ixs:
                    point_i = [i for i,x in enumerate(proto_ixs) if point == x][0]
                    selectedpoints_j.append(point_i)
        fig.add_trace(
            go.Scatter(
                x=x[j], y=y[j], text=classes[j], name=proto_list[j],
                mode='markers',selectedpoints=selectedpoints_j,
                marker={'size': size, 'symbol':'cross', 'color':colors[j%10]}),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=x_centers[j][:s], y=y_centers[j][:s],
                text=classes_centers[j][:s], name=label_list[j],
                selectedpoints=None,mode='markers',
                marker={'size': 5, 'color':colors[j%10], 'opacity':0.6}),
            row=1, col=1
        )
        # if len(selectedpoints) == 0:
        #     fig.add_trace(go.Scatter(x=x[j], y=y[j],text=classes[j], name= label_list[j],mode='markers',marker={'size': size, 'symbol':'cross', 'color':colors[j]}), row=1, col=1)
        #     fig.add_trace(go.Scatter(x=x_centers[j][:s], y=y_centers[j][:s],text=classes_centers[j][:s], name= label_list[j],mode='markers',marker={'size': 6,'color':colors[j],'opacity':0.7}), row=1, col=1)

        # else:
        #     proto_ixs = prototypes_ixs[j]
        #     selectedpoints_j = []
        #     for point in selectedpoints:
        #         if point in proto_ixs:
        #             point_i = [i for i,x in enumerate(proto_ixs) if point == x][0]
        #             selectedpoints_j.append(point_i)
        #     fig.add_trace(go.Scatter(x=x[j], y=y[j],text=classes[j], name=label_list[j],mode='markers',selectedpoints=selectedpoints_j,marker={'size': size, 'symbol':'cross', 'color':colors[j]}), row=1, col=1)
        #     fig.add_trace(go.Scatter(x=x_centers[j][:s], y=y_centers[j][:s],text=classes_centers[j][:s], name=label_list[j],selectedpoints=[],mode='markers',marker={'size': 6,'color':colors[j],'opacity':0.7}), row=1, col=1)
    fig.update_layout()

    components_dict = {0: 'First', 1: 'Second', 2: 'Third', 3: 'Fourth'}

    fig.update_layout(
        title="Prototypes and data instances in the 2D space (PCA)",
        xaxis_title=components_dict[x_select] + " principal component (x)",
        yaxis_title=components_dict[y_select] + " principal component (y)",
        clickmode='event+select',
        uirevision=True,
        width=1000,
        height=600,
    )


    return fig


def generate_figure_weights(model_container, selected=None, label_list=class_names):
    fig_weights = go.Figure(
        px.imshow(model_container.prototypes.W_dense.T,origin='lower'),
        layout=go.Layout(title=go.layout.Title(text="A Bar Chart"))
    )
    fig_weights.update_traces(dict( showscale=False, colorbar_len=0.1,
                       coloraxis=None), selector={'type':'heatmap'})
    #fig_weights.update_traces(showscale=False)

    fig_weights.update_layout(clickmode='event+select')
    if selected is not None:
        fig_weights.add_trace(go.Scatter(x=[selected],y=[1]))

    _,_,_,prototypes_classes,_ = model_container.prototypes.get_all_instances()

    xticks = []
    for j in range(len(label_list)):
        tickj = np.mean(np.argwhere(np.array(prototypes_classes) == j))
        xticks.append(tickj)

    fig_weights.update_layout(
        title="Weights of the last fully-connected layer",
        xaxis_title="Prototypes",
        yaxis_title="Classes",
        #margin = {'l': 10, 'b': 10, 't': 10, 'r': 10},
        xaxis = dict(
            tickmode = 'array',
            tickvals = xticks,
            ticktext = label_list #class_names2
        ),
        yaxis = dict(
            tickmode = 'array',
            tickvals = [i for i in range(len(class_names))],
            ticktext = label_list #class_names
        ),
        width=1000,
        height=300,

    )
    return fig_weights

def generate_figure_mel(mel_spec):
    figure = go.Figure(px.imshow(mel_spec.T,origin='lower'),layout=go.Layout(title=go.layout.Title(text="A Bar Chart")))
    figure.update_traces(dict( showscale=False, colorbar_len=0.1,
                       coloraxis=None), selector={'type':'heatmap'})
    figure.update_layout(
        title="Mel-spectrogram",
        xaxis_title="Time (hops)",
        yaxis_title="Mel filter index",
        #margin = {'l': 0, 'b': 0, 't': 40, 'r': 10}
    )
    #figure.layout.coloraxis.showscale = False
    return figure




class GUI():
    def __init__(self, model_containers, data, folds_files, exp_folder_input, exp_folder_output, label_list, params, plot_label_list=None,graph=None):
        self.model_containers = model_containers
        self.data = data
        self.folds_files = folds_files
        self.exp_folder_input = exp_folder_input
        self.exp_folder_output = exp_folder_output
        self.label_list = label_list
        self.params = params
        if plot_label_list is None:
            self.plot_label_list = label_list
        else:
            self.plot_label_list = plot_label_list
        self.graph = graph
        
        self.fold_list = list(model_containers.keys())
        self.fold_test = self.fold_list[0]
        self.fold_val = get_fold_val(self.fold_test, self.fold_list)

        self.samples_per_class = 10    
        self.x_select = 0    
        self.y_select = 1 
    
        self.click_timestamps = [0,0,0]
        self.generate_figure2D()
        self.generate_figure_weights()
        
                
    def generate_layout(self, app):
        import tensorflow as tf

        external_stylesheets = [
        'https://codepen.io/chriddyp/pen/bWLwgP.css',
        {
            'href': 'https://stackpath.bootstrapcdn.com/bootstrap/4.1.3/css/bootstrap.min.css',
            'rel': 'stylesheet',
            'integrity': 'sha384-MCw98/SFnGE8fJT3GXwEOngsV7Zt27NXFoaoApmYm81iuXoPkFOJwJ8ERdknLPMO',
            'crossorigin': 'anonymous'
        } ]
        self.app = app

        self.graph = tf.get_default_graph()

        self.generate_figure2D()
        self.generate_figure_weights()
        plot2D = dcc.Graph(id='plot2D', figure=self.figure,
                           style={"height" : "100%", "width" : "100%"})

        _,center_mel_blank,_,_,_,_  = self.model_containers[self.fold_test].data_instances.get_instance_by_index(0)
        plot_mel = dcc.Graph(id="plot_mel", 
                          figure = self.generate_figure_mel(center_mel_blank),
                          style={"width": "70%", "display": "inline-block",'float':'left'}
        )
        plot_weights = dcc.Graph(id="plot_weights", 
                                  figure = self.fig_weigths, 
                                  style={"width": "100%", "display": "inline-block"}
        )
        audio = dash_audio_components.DashAudioComponents(id='audio-player', src="",
                                                          autoPlay=False, controls=True)

        button_delete = html.Button('Delete prototype',id='delete_and_convert', className='button',n_clicks_timestamp=0,style={'display':'none','width':'70%'})
        button_eval = html.Button('Evaluate model',id='eval', className='button',n_clicks_timestamp=0,style={'width':'70%'})
        button_load = html.Button('Load best weigths',id='load_weigths', className='button',n_clicks_timestamp=0,style={'width':'70%'})
        button_train = html.Button('Train model',id='train', className='button',n_clicks_timestamp=0,style={'width':'70%'})
        button_reset = html.Button('Reset model',id='reset', className='button',n_clicks_timestamp=0,style={'width':'70%'})
        output_eval = html.Div(id='output_eval',style={'width':'20%'})
        output_text = html.Div(id='output_text')
        output_interval = html.Div(id='output_interval')

        input_epochs =  dcc.Input(id="input_epochs", type="number", placeholder="epochs",min=1, max=100, step=1,style={'width':'33%'})#,value=10)
        input_lr =  dcc.Input(id="learning_rate", type="number", placeholder="learning_rate",min=0.0000001, max=1,style={'width':'33%'})
        input_bs =  dcc.Input(id="batch_size", type="number", placeholder="batch_size",min=32, max=512, step=32,style={'width':'33%'})#,value=64)

        slider_samples = html.Div(dcc.Slider(id='samples_per_class',min=1,max=500, step=1,value=10,vertical=False),style={'width':'100%'})

        interval = dcc.Interval(id='interval-component', interval=1*1000, # in milliseconds
                                n_intervals=0)

        options = []
        for fold in self.fold_list:
            option = {'value': fold, 'label': fold}
            options.append(option)

        fold_select = dcc.Dropdown(id='fold_select',options=options,value=self.fold_test,style={'width':'85%'})
        #model_select = dcc.Dropdown(id='model_select',options=available_models,value=model_input_name,style={'width':'85%'})
        #input_model_output =  dcc.Input(id="input_model_output", type="text", placeholder="model output",style={'width':'70%'},value=model_output_name)#,value=64)

        options = []
        for j in range(4):
            options.append({'label':'component '+str(j+1),'value':j})
        x_select = dcc.Dropdown(id='x_select',options=options,value=0,style={'width': '80%'})
        y_select = dcc.Dropdown(id='y_select',options=options,value=1,style={'width': '80%'})

        eval_div = html.Div([button_eval,output_eval],style={'columnCount': 2,'width':'50%'})
        train_div = html.Div([input_epochs,input_lr,input_bs],style={'width':'70%'})#,style={'columnCount': 4,'width':'80%'})
        model_div = html.Div([fold_select],style={'columnCount': 3,'width':'80%'})
        model_prop_div = html.Div([button_load,button_reset],style={'columnCount': 2,'width':'50%'})                                

        #self.app.layout = html.Div([ html.Div([plot_mel, graph2,plot_weights ], className="nine columns",style={'height':'80vh'}) ])
        self.app.layout = html.Div([
            html.Div([
                html.Div([x_select], className="two columns",style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
                html.Div([y_select], className="two columns",style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
                html.Div([slider_samples], className="three columns",style={'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),
            ], className="row", style={'height':'10vh'}),

            html.Div([
                #html.Div([slider_samples], className="one column"),
                html.Div([plot2D], className="nine columns",style={'height':'80vh'}),

                html.Div([plot_mel,html.Br(),audio,html.Br(),button_delete,html.Br(),button_eval,html.Br(),output_eval,html.Br(),button_load,html.Br(),
                            button_reset,html.Br(),button_train,html.Br(),train_div,html.Br(),
                            html.Br(),fold_select, #model_select,input_model_output
                            html.Br(),html.Br(),output_text,interval], className="three columns"),
            ], className="row", style={'height':'80vh'}),
        #     html.Div([
        #        html.Div([slider_samples], className="nine columns", style={'align':'center'})
        #    ], className="row"),

            html.Div([
            # html.Div([], className="two columns"),
                html.Div([plot_weights], className="six columns"),
                #html.Div([graph_log], className="six columns")
            ], className="row", style={'height':'30vh'})
        ])


    def generate_figure2D(self,selectedpoints=[]):
        prototypes_feat,prototypes_mel,protoypes2D,prototypes_classes,_ = self.model_containers[self.fold_test].prototypes.get_all_instances()

        prototype_ixs = np.arange(0,len(prototypes_feat))
        x = []
        y = []
        classes = []
        classes_ix = []
        prototypes_ixs = []
        for class_ix in range(10):
            prototypes_class_ix = protoypes2D[prototypes_classes == class_ix]
            prototype_ixs_class = prototype_ixs[prototypes_classes == class_ix]
            xj = []
            yj = []
            classesj = []
            for j in range(len(prototypes_class_ix)):
                xj.append(prototypes_class_ix[j,self.x_select])
                yj.append(prototypes_class_ix[j,self.y_select])
                classesj.append('prototype'+str(prototype_ixs_class[j]))
               # classes_ix.append(int(prototypes_classes[j]))
            x.append(xj)
            y.append(yj)
            classes.append(classesj)    
            prototypes_ixs.append(prototype_ixs_class)


        centers_feat,centers_mel,centers2D,centers_classes,centers_audio,centers_file_names = self.model_containers[self.fold_test].data_instances.get_all_instances()
        centers_ixs = np.arange(0,len(centers2D))
        x_centers = []
        y_centers = []
        classes_centers = []
        classes_ix_centers = []

        ### Add this to tests. Delete!!!
        #centers2D = self.model_containers[self.fold_test].data_instances.X_feat_2D['X']#self.X_2D[self.fold_test]['X']
        #centers_classes = self.model_containers[self.fold_test].data_instances.X_feat_2D['Y']
        #centers_ixs = np.arange(0,len(centers2D))

        for class_ix in range(10):
            centers_class_ix = centers2D[centers_classes == class_ix]
            centers_ixs_class = centers_ixs[centers_classes == class_ix]
            xj = []
            yj = []
            classesj = []
            for j in range(len(centers_class_ix)):
                xj.append(centers_class_ix[j,self.x_select])
                yj.append(centers_class_ix[j,self.y_select])
                classesj.append('center'+str(centers_ixs_class[j]))
               # classes_ix.append(int(prototypes_classes[j]))
            x_centers.append(xj)
            y_centers.append(yj)
            classes_centers.append(classesj)  



        fig = make_subplots(rows=1, cols=1)#, column_widths=[0.8, 0.2])
        size = 12
        for j in range(10):
            s = min(self.samples_per_class,len(x_centers[j]))
            print(s,self.samples_per_class,len(x_centers[j]))
            if len(selectedpoints) == 0:
                fig.add_trace(go.Scatter(x=x[j], y=y[j],text=classes[j], name=self.label_list[j],mode='markers',marker={'size': size, 'symbol':'cross', 'color':colors[j]}), row=1, col=1)
                fig.add_trace(go.Scatter(x=x_centers[j][:s], y=y_centers[j][:s],text=classes_centers[j][:s], name=self.label_list[j],mode='markers',marker={'size': 6,'color':colors[j],'opacity':0.7}), row=1, col=1)

            else:
                proto_ixs = prototypes_ixs[j]
                selectedpoints_j = []
                for point in selectedpoints:
                    if point in proto_ixs:
                        print(point,proto_ixs)
                        point_i = [i for i,x in enumerate(proto_ixs) if point == x][0]
                        selectedpoints_j.append(point_i)
                fig.add_trace(go.Scatter(x=x[j], y=y[j],text=classes[j], name=self.label_list[j],mode='markers',selectedpoints=selectedpoints_j,marker={'size': size, 'symbol':'cross', 'color':colors[j]}), row=1, col=1)
                fig.add_trace(go.Scatter(x=x_centers[j][:s], y=y_centers[j][:s],text=classes_centers[j][:s], name=self.label_list[j],selectedpoints=[],mode='markers',marker={'size': 6,'color':colors[j],'opacity':0.7}), row=1, col=1)
        fig.update_layout()

        components_dict = {0: 'First', 1: 'Second', 2: 'Third', 3: 'Fourth'}

        fig.update_layout(
        title="Prototypes and k-means centers in the 2D space (PCA)",
        xaxis_title=components_dict[self.x_select] + " principal component (x)",
        yaxis_title=components_dict[self.y_select] + " principal component (y)",
        clickmode='event+select',uirevision=True
        )
        self.figure = fig
        return fig

    def generate_figure_training(self):
        data = []
        weights_folder = os.path.join(self.weights_folder, self.model_output_name)
        if len(self.training_logs) > 0:
            for j,training_log in enumerate(self.training_logs[self.fold_test]):
                #print(training_log)
                epochs,val_acc,name = training_log['epochs'],training_log['val_acc'],training_log['name']
                if training_log['training'] == True:
                    epochs, val_acc = load_training_log(weights_folder,self.fold_test,row_ix=11)
                if len(epochs) > 0:
                    best = 0
                    for val in val_acc:
                        if float(val)>best:
                            best = float(val)

                    data.append({'x':epochs,'y': val_acc,'name': name,'mode': 'markers','marker': {'size': 8, 'color': colors[j]}}) #'val_acc_'+
                    data.append({'x':[epochs[0],epochs[-1]],'y': [best,best],'name': 'best_'+name,'mode': 'lines','marker': {'color': colors[j]}}) #'best_val_acc_'+


            self.figure_training = go.Figure(data=data)

            self.figure_training.update_layout(
            title="Accuracy on the validation set",
            xaxis_title="Accuracy",
            yaxis_title="Number of epochs",
            clickmode= 'event+select',uirevision=True
            )
        else:
            self.figure_training = go.Figure(data={'x':[0],'y': [0]})

            self.figure_training.update_layout(
            title="Accuracy on the validation set",
            xaxis_title="Accuracy",
            yaxis_title="Number of epochs",
            clickmode= 'event+select',uirevision=True
            )        
        return self.figure_training

    def generate_figure_weights(self,selected=None):
        fig_weigths = go.Figure(px.imshow(self.model_containers[self.fold_test].prototypes.W_dense.T,origin='lower'),layout=go.Layout(title=go.layout.Title(text="A Bar Chart")))
        fig_weigths.update_layout(clickmode='event+select')
        if selected is not None:
            fig_weigths.add_trace(go.Scatter(x=[selected],y=[1]))


        _,_,_,prototypes_classes,_ = self.model_containers[self.fold_test].prototypes.get_all_instances()
        xticks = []
        for j in range(10):
            tickj = np.mean(np.argwhere(np.array(prototypes_classes) == j))
            xticks.append(tickj)
        self.fig_weigths = fig_weigths
        self.fig_weigths.update_layout(
            title="Weights of the last fully-connected layer",
            xaxis_title="Prototypes",
            yaxis_title="Classes",
            #margin = {'l': 0, 'b': 0, 't': 40, 'r': 10}
            xaxis = dict(
                tickmode = 'array',
                tickvals = xticks,
                ticktext = class_names2
            ),
            yaxis = dict(
                tickmode = 'array',
                tickvals = [i for i in range(len(class_names))],
                ticktext = class_names
            )

        )

        return fig_weigths

    def generate_figure_mel(self,mel_spec):
        figure = go.Figure(px.imshow(mel_spec.T,origin='lower'),layout=go.Layout(title=go.layout.Title(text="A Bar Chart")))
        figure.update_layout(
            title="Mel-spectrogram",
            xaxis_title="Time (hops)",
            yaxis_title="Mel filter index",
            #margin = {'l': 0, 'b': 0, 't': 40, 'r': 10}
        )
        #figure.layout.coloraxis.showscale = False
        return figure

    def add_mel_to_figure(self,hoverData):
        point = np.array([hoverData['points'][0]['x'],hoverData['points'][0]['y']])

        dist_protos = self.model_containers[self.fold_test].prototypes.get_distances(point,components=(self.x_select,self.y_select))
        dist_data = self.model_containers[self.fold_test].data_instances.get_distances(point,components=(self.x_select,self.y_select))
        #print(np.amin(dist_data),np.amin(dist_protos))
        if np.amin(dist_data) <= np.amin(dist_protos): # click on k-mean
            arg_dist = np.argmin(dist_data)  
          #  print(arg_dist)
            (center_mel,center_feat,center_2D,
            center_class,center_file,center_audio)=self.model_containers[self.fold_test].data_instances.get_center(arg_dist)  
            from PIL import Image

            image_array = np.random.randint(0, 255, size=(100, 100)).astype('uint8')
            center_mel = cm(center_mel.T)
            #center_mel = 255*(center_mel-np.amin(center_mel))/(np.amax(center_mel)-np.amin(center_mel))
            image = Image.fromarray((center_mel[:, :, :3] * 255).astype('uint8'))



            layout= go.Layout(images= [dict(
                              source= image,
                              xref= "x",
                              yref= "y",
                              x= hoverData['points'][0]['x']-0.5,
                              y= hoverData['points'][0]['y']+2,
                              sizex= 2,
                              sizey= 2,
                              #sizing= "stretch",
                              opacity= 1.0#,layer= "below"
                              )])
            self.figure.update_layout(layout)

        else:
            arg_dist = np.argmin(dist_protos)
            (proto_feat,proto_mel,
            proto_2D,proto_class,proto_audio) = self.model_containers[self.fold_test].prototypes.get_prototype_by_index(arg_dist)



    def display_plot(self,clickData):
        temp_mel = np.ones((64,128))
        if isinstance(clickData,dict):
            point = np.array([clickData['points'][0]['x'],clickData['points'][0]['y']])
            print(self.fold_test)
            dist_protos = self.model_containers[self.fold_test].prototypes.get_distances(point,components=(self.x_select,self.y_select))
            dist_data = self.model_containers[self.fold_test].data_instances.get_distances(point,components=(self.x_select,self.y_select))
            if np.amin(dist_data) <= np.amin(dist_protos): # click on k-mean
                arg_dist = np.argmin(dist_data)  
                (center_feat,center_mel,center_2D,
                center_class,center_audio,center_file)=self.model_containers[self.fold_test].data_instances.get_instance_by_index(arg_dist)  
                self.selected = {'type': 'center', 'id': arg_dist}
                
                figure = self.generate_figure_mel(center_mel)
                
                data, sr = librosa.core.load(center_file)
                return [figure, 
                        {'autoPlay': True, 'src': encode_audio(data,sr)}, #encode_audio(center_audio['data'],center_audio['sr'])
                        "Convert center to Prototype", {'display':'inline-block','width':'70%'}]
            else:
                arg_dist = np.argmin(dist_protos)
                (proto_feat,proto_mel,
                proto_2D,proto_class,proto_audio) = self.model_containers[self.fold_test].prototypes.get_instance_by_index(arg_dist)
                self.selected = {'type': 'prototype', 'id': arg_dist}
                figure = self.generate_figure_mel(proto_mel)
           
                return [figure, 
                        {'autoPlay': True, 'src': encode_audio(proto_audio['data'],proto_audio['sr'])},
                        "Delete Prototype", {'display':'inline-block','width':'70%'}]
        else:
            return [self.generate_figure_mel(temp_mel), {'autoPlay': False, 'src': ''},
                    "Select a point", {'display':'none','width':'70%'}]

    def buttons_and_others(self,btn1,btn2,btn3,fold_selected,clickData,samples_per_class,x_select,y_select,epochs,learning_rate,batch_size,selectedData,selectedData_w):
        if x_select != self.x_select:
            self.x_select = x_select
            self.generate_figure2D()
            return [self.figure, self.fig_weigths]
        if y_select != self.y_select:
            self.y_select = y_select
            self.generate_figure2D()
            return [self.figure, self.fig_weigths]

        if samples_per_class != self.samples_per_class:
            #print(samples_per_class,self.samples_per_class)
            self.samples_per_class = samples_per_class
            self.generate_figure2D()
            #print('new figure')
            return [self.figure, self.fig_weigths]

        # if model_select != self.model_input_name:
        #     print(model_select,self.model_input_name)
        #     self.model_input_name = model_select
        #     scaler_path = os.path.join(scaler_folder, 'base')
        #     self.load_model_prototypes_centers(folds_data_test,folds_files,scaler_path)
        #     return [self.figure, self.fig_weigths]
        #print(clickData,selectedData_w)
                
        #self.generate_figure2D(selectedpoints=clickData)
        #print(fold_selected,self.fold_test)
        if fold_selected != self.fold_test:
            
            self.change_fold(fold_selected)
            return [self.figure, self.fig_weigths]

        #print(clickData)
        if clickData is not None:
            selected_prototype = clickData['points'][0]['x']
            #print(selected_prototype,self.selected_prototype)
            if selected_prototype != self.selected_prototype:
                self.selected_prototype = selected_prototype
                self.generate_figure2D([selected_prototype])
                #print(clickData)
                return [self.figure, self.fig_weigths]
        #print(btn1,btn2,btn3,self.click_timestamps[0],self.click_timestamps[1],self.click_timestamps[2])

        if int(btn1) > self.click_timestamps[0]:
            self.click_timestamps[0] = int(btn1)
            self.click_delete(selectedData)
            
        if int(btn2) > self.click_timestamps[1]:
            self.click_timestamps[1] = int(btn2)
            self.click_reset()
            
        if int(btn3) > self.click_timestamps[2]:
            self.click_timestamps[2] = int(btn3)
            msg = 'Button 3 was most recently clicked'
            if epochs is not None:
                self.params['train']['epochs'] = int(epochs)
            if learning_rate is not None:
                self.params['train']['learning_rate'] = learning_rate
            if batch_size is not None:
                self.params['train']['batch_size'] = int(batch_size)
            print(epochs,learning_rate,batch_size)
            self.train_model()
            #self.model_output_name = model_output_name
            #scaler_path = os.path.join(scaler_folder, 'base')
            #weights_folder_debug_manual = os.path.join(weights_folder, 'debug_manual2')
            #last_training_log = self.get_training_log()[-1]
            #initial_epoch = int(last_training_log['epochs'][-1])+1
            #self.train_model(folds_data=folds_data,folds_data_test=folds_data_test,folds_files=folds_files,scaler_path=scaler_path,
            #            epochs=epochs,learning_rate=learning_rate,batch_size=batch_size,fit_verbose=1,convert_audio_dict=convert_audio_dict,graph=graph,initial_epoch=initial_epoch)

        return [self.figure, self.fig_weigths]


    def btn_load(self,n_clicks_timestamp,n_clicks_timestamp2):
        if n_clicks_timestamp > n_clicks_timestamp2:
            #self.load_weights(weights_folder_debug_manual)
            return "TODO"#"Weights loaded from " + weights_folder_debug_manual
        elif n_clicks_timestamp2 > n_clicks_timestamp:
            acc = self.eval_model()
            return "Accuracy in fold {:s}: {:f}".format(self.fold_val, acc)
        else:
            return ""

    def click_delete(self,selectedData):
        #msg = 'Button 1 was most recently clicked'
        point = np.array([selectedData['points'][0]['x'],selectedData['points'][0]['y']])
        dist_protos = self.model_containers[self.fold_test].prototypes.get_distances(point,components=(self.x_select,self.y_select))
        dist_data = self.model_containers[self.fold_test].data_instances.get_distances(point,components=(self.x_select,self.y_select))
        #print(np.amin(dist_data),np.amin(dist_protos))
        if np.amin(dist_data) <= np.amin(dist_protos): # click on k-mean
            arg_dist = np.argmin(dist_data)    
            (center_feat,center_mel,center_2D,
             center_class,_,center_file) = self.model_containers[self.fold_test].data_instances.remove_instance(arg_dist)
            data, sr = librosa.core.load(center_file)  
            center_audio = {'data':data, 'sr': sr}          
            self.model_containers[self.fold_test].prototypes.add_instance(int(center_class),
                                                center_mel,center_feat,
                                                embedding2D=center_2D,audio=center_audio)  

        else:
            arg_dist = np.argmin(dist_protos)
            self.model_containers[self.fold_test].prototypes.remove_instance(arg_dist) 

        self.generate_figure2D()
        self.generate_figure_weights()
        return self.figure

    def click_reset(self):
        self.model_containers[self.fold_test].data_instances.reset()
        self.model_containers[self.fold_test].prototypes.reset()
        self.generate_figure2D()
        self.generate_figure_weights()
        return self.figure

    def change_fold(self,fold_selected):
        print('fold_selected', fold_selected)
        self.fold_test = fold_selected
        self.fold_val = get_fold_val(self.fold_test, self.fold_list)
        print(self.fold_val)
        self.generate_figure2D()
        #self.generate_figure_training()
        self.generate_figure_weights()
        return self.figure

    def eval_model(self):

        with self.graph.as_default():
            self.update_model_to_prototypes()
            scaler_path = os.path.join(self.exp_folder_input, self.fold_test, 'scaler.pickle')
            scaler = load_pickle(scaler_path)
            acc,_,_ = self.model_containers[self.fold_test].evaluate(self.data[self.fold_val]['X'],self.data[self.fold_val]['Y'], scaler)
            return acc

    def update_model_to_prototypes(self):
        N_protos = self.model_containers[self.fold_test].prototypes.get_number_of_instances()
        n_classes = len(self.label_list)
        #self.model_containers[self.fold_test].model = debugg_model(self.model_containers[self.fold_test].model,N_protos,n_classes)
        
        #self.model_containers[self.fold_test].model.get_layer('prototype_distances').set_weights([self.model_containers[self.fold_test].prototypes.embeddings])
        #self.model_containers[self.fold_test].model.get_layer('mean').set_weights([self.model_containers[self.fold_test].prototypes.W_mean])
        #self.model_containers[self.fold_test].model.get_layer('logits').set_weights([self.model_containers[self.fold_test].prototypes.W_dense])
        n_frames_cnn,n_freq_cnn = self.model_containers[self.fold_test].prototypes.mel_spectrograms[0].shape
        
        N_filters_last = self.model_containers[self.fold_test].model.get_layer('features').output_shape[-1]

        model = modelAPNet(n_prototypes=N_protos, n_frames_cnn=n_frames_cnn, n_freq_cnn=n_freq_cnn, N_filters=[16,16,N_filters_last])

        for layer in model.layers:
            if len(layer.get_weights()) > 0: 
                if layer.name == 'prototype_distances':
                    model.get_layer(layer.name).set_weights([self.model_containers[self.fold_test].prototypes.embeddings])
                elif layer.name == 'mean':
                    model.get_layer(layer.name).set_weights([self.model_containers[self.fold_test].prototypes.W_mean])
                elif layer.name == 'logits':
                    model.get_layer(layer.name).set_weights([self.model_containers[self.fold_test].prototypes.W_dense])
                elif layer.name == 'input':
                    continue
                else:
                    model.get_layer(layer.name).set_weights(self.model_containers[self.fold_test].model.get_layer(layer.name).get_weights())
        
        self.model_containers[self.fold_test].model = model

    def train_model(self):

        with self.graph.as_default():
            self.update_model_to_prototypes()

            # paths
            dataset = self.exp_folder_output.split("/")[-1] #TODO fix this
            exp_folder_fold = os.path.join(self.exp_folder_output, self.fold_test)

            weights_path = exp_folder_fold#os.path.join(exp_folder_fold, 'best_weights.hdf5') 
            log_path = os.path.join(exp_folder_fold, 'training.log') 
            scaler_path = os.path.join(self.exp_folder_input, self.fold_test, 'scaler.pickle') 

            params_model = self.params["models"]['APNet']
            params_dataset = self.params["datasets"][dataset]
            kwargs = self.params["train"]
            if 'train_arguments' in params_model:
                kwargs.update(params_model['train_arguments'])
            kwargs.update({'init_last_layer': False})

            # save model as json
            self.model_containers[self.fold_test].save_model_json(exp_folder_fold)

            X_train, Y_train, X_val, Y_val = get_data_train(self.data, self.fold_test, params_dataset["evaluation_mode"])
            # HERE HAS TO BE DATA FOR TRAINING
            
            #print(X_train.shape,Y_train.shape,X_val.shape,Y_val.shape)
            scaler = load_pickle(scaler_path)

            X_train = scaler.transform(X_train)
            X_val = scaler.transform(X_val)
            
            self.model_containers[self.fold_test].train(X_train, Y_train, X_val, Y_val, 
                                                  weights_path=weights_path, log_path=log_path, **kwargs)
            # load best_weights after training                                      
            self.model_containers[self.fold_test].model.load_weights(os.path.join(exp_folder_fold, 'best_weights.hdf5'))
        #     val_out_acc = history.history['val_out_acc']
        #     epochs = [i for i in range(initial_epoch,epochs+initial_epoch)]
        #     self.training_logs[self.fold_test][-1]['epochs'] = epochs
        #     self.training_logs[self.fold_test][-1]['val_acc'] = val_out_acc
        #     self.training_logs[self.fold_test][-1]['training'] = False

        #     #print(self.training_logs[self.fold_test][-1])
        #     #print(history.history)
            print('Reloading the plot')  
            data_instances_path = os.path.join(exp_folder_fold, 'data_instances.pickle')
            prototypes_path = os.path.join(exp_folder_fold, 'prototypes.pickle') 
            X_feat,X_train,Y_train,Files_names_train = get_data_test(self.model_containers[self.fold_test].model,self.data,self.fold_test,self.folds_files,scaler)
            # TODO: data_centers[fold_test] = Data_centers(X_feat,X_train,Y_train,Files_names_train,n_classes=10,n_clusters=5)
            mel_basis = np.load(os.path.join(params_dataset['feature_folder'], 'mel_basis.npy'))
            convert_audio_params = {'sr': self.params['features']['sr'], 
                                    'scaler' : scaler, 
                                    'mel_basis' : mel_basis, 
                                    'audio_hop' : self.params['features']['audio_hop'], 
                                    'audio_win' : self.params['features']['audio_win']}
            projection2D = self.model_containers[self.fold_test].data_instances.projection2D
            self.model_containers[self.fold_test].get_prototypes(X_train, 
                                                        convert_audio_params=convert_audio_params, 
                                                        projection2D=projection2D)
            data_instances_path = os.path.join(self.exp_folder_output, 'data_instances.pickle')
            prototypes_path = os.path.join(self.exp_folder_output, 'prototypes.pickle') 
            save_pickle(self.model_containers[self.fold_test].data_instances, data_instances_path)
            save_pickle(self.model_containers[self.fold_test].prototypes, prototypes_path)
            self.generate_figure2D()
            self.generate_figure_weights()
            return self.figure

    def load_weights(self, weights_folder=''): 
        
        weights_file = 'fold' + str(self.fold_test) + '.hdf5' #_{epoch:02d}
        weights_path = os.path.join(weights_folder, weights_file)   

        self.model_containers[self.fold_test].model.load_weights(weights_path)

        #scaler = load_scaler(scaler_path,self.fold_test)


    def get_training_log(self):
        return self.training_logs[self.fold_test]

    def append_training_log(self,training_log_new):
        self.training_logs[self.fold_test].append(training_log_new)
     
