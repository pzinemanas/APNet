#from .model import get_features,get_classes,get_prototypes
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA, IncrementalPCA
from librosa import db_to_power
from librosa.util import nnls
from librosa.core import griffinlim
import os
import soundfile as sf
import time
from shutil import copyfile
from operator import itemgetter

from dcase_models.util import progressbar


class DataRepresentations:
    """
    A super class to store different forms of the representations of
    data utils for APNet model. These representations are mel-spectrograms,
    embeddigns, embeddigns2D, classes (annotations) and audio signals.

    Attributes
    ----------
    mel_spectrograms : ndarray
        3D array with the mel-spectrograms.
        Shape = (N_instances, N_hops, N_mel_bands)
    embeddings : ndarray
        4D array with the embeddings.
        Shape (N_instances, N_hops_feat, N_freqs_feat, N_filters)
    classes : ndarray
        1D array with the classses of each instance.
        Shape (N_instances,)
    convert_audio_params : dict
        Dictionary with parameters needed for audio extraction
        keys: 'sr', 'scaler', 'mel_basis', 'audio_hop', 'audio_win'}
    audios : list of dicts
        list (len of N_instances) of dicts with keys {'data', 'sr'}, 
        where 'data' is the audio samples and 'sr' is the sampling rate
    projection2D : sklearn.decomposition, optional
        sklean Object to project the embedding space into a 2D space.
    embeddings2D : ndarray
        2D array with the embedding space transform to a 2D space.
        Shape = (N_instances, 2)
    originals : dict
        Dictionary with the attribute values when the instance was init.

    Methods
    -------
    sort()
        Sort all the representations by class numbers (self.classes)
    remove_instance(index)
        Remove the instance given by the index
    add_instance(class_ix, mel_spectrogram, embedding, embedding2D=None, audio=None)
        Add an instance of the data in all the representations
    convert_to_audio()
        Generate self.audios with the data of self.mel_spectrograms and the
        parameters given by self.convert_audio_params
    save_audio_files(path)
        Save self.audios signals to files in the given path
    get_distances(point, classes=None, components=(0,1))
        Returns the distance of point to each instance in self.embeddings2D.
        Use in visualization and manual debugging
    get_all_instances()
        Returns all instances of all representations
    get_instance_by_index(index)
        Returns the representations for the instance given by index
    get_number_of_instances()
        Get number of intances, N_instances
    reset()
        Restore the attributes values saved in self.originals 
    """

    def __init__(self, mel_spectrograms, embeddings, classes, 
                 projection2D=None, convert_audio_params=None):
        """
        Parameters
        ----------
        mel_spectrograms : ndarray
            3D array with mel-spectrograms of the instances.
            Shape = (N_instances, N_hops, N_mel_bands)
        embeddings : ndarray
            4D array with the embeddings of the instances.
            Shape (N_instances, N_hops_feat, N_freqs_feat, N_filters)
        classes : ndarray
            1D array with the annotations of instances.
            Shape (N_instances, )
        projection2D : sklearn.decomposition, optional
            sklean Object to project the embedding space into a 2D space.
            If is None, self.embeddings2D is set to None
        convert_audio_params : dict, optional
            Dictionary with parameters needed for audio extraction
            keys: 'sr', 'scaler', 'mel_basis', 'audio_hop', 'audio_win'}  
            If is None, self.audios is set to None  
        """
        self.mel_spectrograms = mel_spectrograms
        self.embeddings = embeddings
        self.classes = classes
        self.convert_audio_params = convert_audio_params
        
        self.projection2D = projection2D

        if convert_audio_params is not None:
            self.convert_to_audio()
        else:
            self.audios = np.zeros(len(self.embeddings))

        if self.projection2D is not None:
            embeddings_flat = np.reshape(self.embeddings,(len(self.embeddings),-1))
            self.embeddings2D = self.projection2D.transform(embeddings_flat)
        else:
            #self.embeddings2D = None
            self.embeddings2D = np.zeros(len(self.embeddings))
        #self.sort() #ACTHUNG! 
        
        self.originals = {'mel_spectrograms':self.mel_spectrograms.copy(),'embeddings' :self.embeddings.copy(),
                         'embeddings2D': self.embeddings2D.copy(), 'classes': self.classes.copy(),
                          'projection2D': self.projection2D, 'audios':self.audios.copy()}

    def sort(self):
        """
        Sort all the representations by class numbers (self.classes)
        """
        sort = np.argsort(self.classes)
        self.classes = self.classes[sort]
        self.mel_spectrograms = self.mel_spectrograms[sort]
        self.embeddings = self.embeddings[sort]
       
        if self.projection2D is not None:
            self.embeddings2D = self.embeddings2D[sort]
        
        if self.convert_audio_params is not None:        
            # TODO make this in a better way
            audios_sorted = []
            for j in range(len(self.audios)):
                audios_sorted.append(self.audios[sort[j]])
                
            self.audios = audios_sorted.copy()
            
    def remove_instance(self, index):
        """
        Remove the instance given by the index

        Parameters
        ----------
        index : int
            Index of the instance to be deleted
        """
        self.classes = np.delete(self.classes,index,axis=0)
        self.mel_spectrograms = np.delete(self.mel_spectrograms,index,axis=0)
        self.embeddings = np.delete(self.embeddings,index,axis=0)
        if self.projection2D is not None:
            self.embeddings2D = np.delete(self.embeddings2D,index,axis=0)
        
        if self.convert_audio_params is not None:
            self.audios.pop(index)
            
        #self.sort()
        
    def add_instance(self, class_ix, mel_spectrogram, embedding, embedding2D=None, audio=None):
        """
        Add an instance of the data in all the representations. The new instance
        is append in the representations and then self.sort() is called.

        Parameters
        ----------
        class_ix : int
            Class of the new instance 
        mel_spectrogram : ndarray
            2D array with mel-spectrogram of the new instance.
            Shape = (N_hops, N_mel_bands)
        embedding : ndarray
            3D array with the embeddings of the new instance.
            Shape (N_hops_feat, N_freqs_feat, N_filters) 
        embedding2D : ndarray or None, optional
            1D array with the coordinates of the new instance in the 2D space
        audio : dict or None, optional 
            Dict with audio signal and sampling rate {'data', 'sr'} of the
            new instance        
        """       
        classes_old = self.classes
        self.classes = np.concatenate((self.classes,np.ones((1,))*class_ix),axis=0).astype(int)
        self.mel_spectrograms = np.concatenate((self.mel_spectrograms,np.expand_dims(mel_spectrogram,axis=0)),axis=0)
        self.embeddings = np.concatenate((self.embeddings,np.expand_dims(embedding,axis=0)),axis=0)
        
        if (embedding2D is None) & (self.projection2D is not None):
            embedding_flat = np.reshape(prototyembeddingpe,(1,-1))
            embedding2D = self.projection2D.transform(prototype_flat)
        
        self.embeddings2D = np.concatenate((self.embeddings2D,np.expand_dims(embedding2D,axis=0)),axis=0)
        
        if audio is not None:
            self.audios.append(audio)

        #self.sort()
       
    def convert_to_audio(self): #path, sr, scaler, mel_basis, audio_hop, audio_win
        """
        Generate self.audios with the data of self.mel_spectrograms and the
        parameters given by self.convert_audio_params.    
        """      
        audios = []
        for j in progressbar(range(len(self.mel_spectrograms))):
            #print(np.amin(self.mel_spectrograms[j]), np.amax(self.mel_spectrograms[j]))
            melspec = self.convert_audio_params['scaler'].inverse_transform(self.mel_spectrograms[j].T)
            #print(np.amin(melspec), np.amax(melspec))
         #   audio_save = mel_spec_to_audio(melspec, self.convert_audio_params['mel_basis'], 
         #                                  self.convert_audio_params['audio_hop'], 
         #                                  self.convert_audio_params['audio_win'])

            melspec = db_to_power(melspec)
            inverse = nnls(self.convert_audio_params['mel_basis'], melspec)
            spec = np.power(inverse, 1./2.0, out=inverse)

            audio = griffinlim(spec, hop_length=self.convert_audio_params['audio_hop'], 
                               win_length=self.convert_audio_params['audio_win'],
                               center=True)

            audio_save = audio #audio[200:-200]   
            audios.append({'data':audio_save,'sr':self.convert_audio_params['sr']})
            
        self.audios = audios        

    def save_audio_files(self, path):
        """
        Save self.audios signals to files in the given path.
        The name of each file is the index instance.

        Parameters
        ----------
        path : str
            Path where the audio signals are saved   
        """     
        for j in range(len(self.audios)):
            
            audio_save = self.audios[j]['data']
            
            file_name = str(j) + '.wav'
            file_path = os.path.join(path,file_name)
            
            sf.write(file_path, audio_save/np.amax(audio_save), self.audios[j]['sr'])            

    def get_distances(self, point, classes=None, components=(0,1)):
        """
        Returns the distance of point to each instance in self.embeddings2D.
        Use in visualization and manual debugging

        Parameters
        ----------
        point : ndarray, tupple, list
            2D coordinates of the point
        classes : list or None, optional
            if is not None, the distance to the instances of these classes are
            set to inf.
        components: tupple or list or ndarray, optional
            Components to used to calculate the distance. Default, use first
            and second components. To use higher components, is necesary to
            have these components in the self.embeddings2D.

        Returns
        -------
        ndarray
            distances of point to all instances in the self.embeddings2D 

        """    
        distances = np.sum(np.power(self.embeddings2D[:,components]-point,2),axis=-1)
        if classes is not None:
            delete = []
            for j in range(len(distances)):
                if self.classes[j] not in classes:      
                    distances[j] = np.inf
        return distances

    def get_all_instances(self):
        """
        Returns all instances of all representations

        Returns
        -------
        ndarray
            embeddings of all instances
        ndarray
            mel_spectrograms of all instances           
        ndarray
            embeddings2D of all instances
        ndarray
            classes of all instances
        list
            audios of all instances
        """    

        return (self.embeddings,self.mel_spectrograms,self.embeddings2D,
                self.classes, self.audios)
    
    def get_instance_by_index(self,index):
        """
        Returns the representations for the instance given by index
        
        Parameters
        ----------
        index : int
            index of the instace to be returned

        Returns
        -------
        ndarray
            embeddings of instance given by index
        ndarray
            mel_spectrogram of instance given by index           
        ndarray
            embeddings2D of instance given by index
        ndarray
            class of instance given by index
        dict
            audio of instance given by index
        """    
        return (self.embeddings[index], self.mel_spectrograms[index], self.embeddings2D[index],
                self.classes[index], self.audios[index])
    
    def get_number_of_instances(self):
        """
        Get number of intances, N_instances
        
        Returns
        ----------
        int
            Number of instances
        """

        return len(self.embeddings)
    
    def reset(self):
        """
        Restore the attributes values saved in self.originals
        """
        
        self.embeddings = self.originals['embeddings'].copy()
        self.mel_spectrograms = self.originals['mel_spectrograms'].copy()   
        self.embeddings2D = self.originals['embeddings2D'].copy()
        self.classes = self.originals['classes'].copy()           
        self.projection2D = self.originals['projection2D']
        self.audios = self.originals['audios'].copy()


class Prototypes(DataRepresentations):
    """
    A child class of DataRepresentations to store different forms of 
    the representations of the prototypes of the APNet model. 
    These representations are mel-spectrograms, embeddigns, 
    embeddigns2D, classes (annotations) and audio signals.

    Attributes
    ----------
    W_dense : ndarray
        2D array with weights of the last dense layer of APNet model
        Shape = (N_prototypes, N_classes)
    W_mean : ndarray
        2 array with weights of the Mean layer of APNet model.
        Shape (N_prototypes, N_freqs_feat)
    prototypes_distances : ndarray
        Matrix distances of prototypes. Distance from each prototype to
        each prototype.

    Methods
    -------
    get_weights(return_classes=False)
        Returns self.W_dense and self.W_mean

    """
    def __init__(self, model_container, X_train, projection2D=None, convert_audio_params=None, random_masks=False):
        model = model_container.model

        model_classes = model_container.model_embeddings_to_out()
        model_features = model_container.model_input_to_embeddings()
        model_distances = model_container.model_input_to_distances() 
        model_prototypes = model_container.model_embeddings_to_decoded()

        embeddings = model.get_layer('prototype_distances').get_weights()[0]

        print('Getting prototypes (spectrograms)...')
        classes, prototypes_distances = model_classes.predict(embeddings)
        classes = np.argmax(classes, axis=1)

        similarity = model_distances.predict(X_train)
        
        argmax = np.argmax(similarity, axis=0)
        _, mask1, mask2 = model_features.predict(X_train[argmax])
        if random_masks:
            mask1 = np.random.binomial(1, 0.25, size=mask1.shape)
            mask2 = np.random.binomial(1, 0.25, size=mask2.shape)
        mel_spectrograms = model_prototypes.predict([embeddings,mask1,mask2])
        print('Done!')

        self.W_dense = model.get_layer('logits').get_weights()[0]
        self.W_mean = model.get_layer('mean').get_weights()[0]
        self.prototypes_distances = prototypes_distances

        print('Converting to audio...')
        super().__init__(mel_spectrograms, embeddings, classes, 
                 projection2D = projection2D, convert_audio_params = convert_audio_params)        
        print('Done!')

        self.originals['W_dense'] = self.W_dense
        self.originals['W_mean'] = self.W_mean

        self.sort()

    def sort(self):
        sort = np.argsort(self.classes)
        self.prototypes_distances = self.prototypes_distances[sort]
        self.prototypes_distances = self.prototypes_distances[:,sort]
        
        self.W_dense = self.W_dense[sort]  
        self.W_mean = self.W_mean[sort]

        super().sort()
            
    def remove_instance(self, index):
        classes_old = self.classes.copy()
        W_dense_old = self.W_dense.copy()
        super().remove_instance(index)
        self.prototypes_distances = np.delete(self.prototypes_distances,index,axis=0)
        self.prototypes_distances = np.delete(self.prototypes_distances,index,axis=1)
        
        self.W_dense = np.delete(self.W_dense,index,axis=0) 
        self.W_mean = np.delete(self.W_mean,index,axis=0)

        j = classes_old[index]
        if np.sum(self.classes==j) >= 1:
            norm_orig = np.sum(np.abs(W_dense_old[classes_old==j]),axis=0,keepdims=True)
            norm_new = np.sum(np.abs(self.W_dense[self.classes==j]),axis=0,keepdims=True)
            self.W_dense[self.classes==j] = norm_orig*self.W_dense[self.classes==j]/norm_new            
        
        self.sort()

    def add_instance(self, class_ix, mel_spectrogram, embedding,
                     embedding2D=None, audio=None, w_dense=None, w_mean=None):    
        classes_old = self.classes
        
        super().add_instance(class_ix, mel_spectrogram, embedding, embedding2D=embedding2D, audio=audio)

        if w_dense is None:
            w_dense = np.ones((1,self.W_dense.shape[1]))*np.mean(self.W_dense[self.W_dense<0])
            w_dense[0,int(class_ix)] = np.mean(self.W_dense[self.W_dense>0])
        
        W_dense_old = self.W_dense
        self.W_dense = np.concatenate((self.W_dense,w_dense),axis=0) 
        
        j = int(class_ix)
        if np.sum(self.classes==j) >= 1:
            norm_orig = np.sum(np.abs(W_dense_old[classes_old==j]),axis=0,keepdims=True)
            norm_new = np.sum(np.abs(self.W_dense[self.classes==j]),axis=0,keepdims=True)
            self.W_dense[self.classes==j] = norm_orig*self.W_dense[self.classes==j]/norm_new 

        if w_mean is None:
            w_mean = np.ones((1,self.W_mean.shape[1]))*np.mean(self.W_mean)
          
        self.W_mean = np.concatenate((self.W_mean,w_mean),axis=0)
        
        #TODO fix this
        n_protos = self.get_number_of_instances()
        self.prototypes_distances = np.zeros((n_protos,n_protos))

        self.sort()
        
    def get_weights(self,return_classes=False):
        if return_classes:
            return self.W_dense,self.W_mean,self.classes
        else:
            return self.W_dense,self.W_mean

    def reset(self): 
        self.W_dense = self.originals['W_dense'].copy()
        self.W_mean = self.originals['W_mean'].copy()           
        super().reset()


class DataInstances(DataRepresentations):
    """
    A child class of DataRepresentations to store different forms of 
    the representations of the training/validation/test set.
    These representations are mel-spectrograms, embeddigns, 
    embeddigns2D, classes (annotations) and audio signals.

    Attributes
    ----------
    file_names
        List of paths for each file

    Methods
    -------
    load_audio()
        Loads the audio signals for each file in self.file_names


    """
    def __init__(self, embeddings, mel_spectrograms, Y, file_names, n_classes=10, use_kmeans = False, n_clusters=5,
                 projection2D=None):
        embeddings_flat = np.reshape(embeddings,(len(embeddings),-1))
        classes = np.argmax(Y,axis=1)

        if projection2D is None:
            print('Fitting PCA...')
            projection2D = PCA(n_components=4)
            #projection2D = IncrementalPCA(n_components=4)
            projection2D.fit(embeddings_flat)
            print('Done!')
            
        if use_kmeans:
            centers = []
            centers_2D = []
            centers_mel = []
            center_classes = []
            centers_file_names = []
            embeddings2D = projection2D.transform(embeddings_flat)
            for j in range(n_classes):
                kmeans = KMeans(n_clusters=n_clusters)
                embeddings_class_j = embeddings[classes == j]
                embeddings2D_class_j = embeddings2D[classes == j] 
                indexes_of_class_j = np.arange(len(embeddings2D))[classes==j] 
                
                kmeans.fit(embeddings2D_class_j)
                
                distances_to_clusters = kmeans.transform(embeddings2D_class_j)
                
                min_dist = np.argmin(distances_to_clusters,axis=0)
                centersj = embeddings_class_j[min_dist]
                centersj_2D = embeddings2D_class_j[min_dist]
                centersj_mel = mel_spectrograms[classes==j][min_dist]
                
                centers_mel.append(centersj_mel)
                centers.append(centersj)
                centers_2D.append(centersj_2D)
                
                center_classes.append(np.ones(n_clusters)*j)
                for k in range(len(min_dist)):
                    centers_file_names.append(files_names[indexes_of_class_j[min_dist[k]]])

        
            embeddings = np.asarray(centers)
            mel_spectrograms = np.asarray(centers_mel)
            classes = np.concatenate(center_classes).astype(int)
           # centers_pca_flatten = np.reshape(centers_pca,(centers_pca.shape[0]*centers_pca.shape[1],
           #                                             centers_pca.shape[2]))
           # centers_flatten = np.reshape(centers_feat,(centers_feat.shape[0]*centers_feat.shape[1],
           #                                             centers_feat.shape[2],centers_feat.shape[3],centers_feat.shape[4]))
           # centers_mel_flatten = np.reshape(centers_mel,(centers_mel.shape[0]*centers_mel.shape[1],
           #                                             centers_mel.shape[2],centers_mel.shape[3]))
        
        super().__init__(mel_spectrograms, embeddings, classes, 
                         projection2D = projection2D, convert_audio_params = None)
        self.file_names = file_names
        #self.load_audio()
        self.originals['file_names'] = file_names.copy() 
        #self.originals['audios'] = self.audios
       
    def remove_instance(self,index):
        deleted = self.get_instance_by_index(index)
        
        super().remove_instance(index)
        self.file_names.pop(index)

        return deleted

    def get_instance_by_index(self,index):
        super_return = super().get_instance_by_index(index)
        return super_return + (self.file_names[index], )
 
    def get_all_instances(self):
        super_return = super().get_all_instances()
        return super_return + (self.file_names, )
    
    def reset(self):
        super().reset()
        self.file_names = self.originals['file_names'].copy()        
  
 #   def convert_to_audio_files(self, path):
 #       for j in range(len(self.centers_mel)):
 #           file_name = 'center' + str(j) + '.wav'
 #           file_path = os.path.join(path,file_name)            
 #           copyfile(self.centers_file_names[j], file_path)

    def load_audio(self):
        self.audios = []
        for j in range(len(self.embeddings)):
            data, sr = sf.read(self.file_names[j])
            self.audios.append({'data':data,'sr': sr})
