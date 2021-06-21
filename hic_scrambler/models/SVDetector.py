# Trying a simple keras NN to predict SVs in a Hi-C matrix.
from os.path import join
import glob
import numpy as np
from numpy import inf
import pandas as pd
from time import time
import matplotlib.pyplot as plt

import os # To remove warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

import keras
from keras.layers import  Input, Conv2D, BatchNormalization, LeakyReLU, MaxPooling2D, Flatten, Dense, Dropout, Add, AveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import  EarlyStopping,ReduceLROnPlateau
from keras.models import model_from_json

from skimage.filters import gaussian, median
import joblib

import warnings
warnings.filterwarnings("ignore")

class SVDetector(object):
    
    """
    Handles to detect SV on a dataset with pictures of HiC matrix. A CNN will 
    be used for this detection. Firstly, we must train the model with "x.npy" 
    and "y.npy" which represent pictures and label. After that we can test the 
    model on new data.
    
    Examples
    --------
        Detector = SVDetector()
        Detector.train()
        Detector.test()
    
    Attributes
    ----------
    n_neurons : int
        Number of neurons in the two last layers of the CNN we will create.
        
    training_path : str
        Path to the npy files to load to create the training dataset.
    """
    
    
    def __init__(self, n_neurons : int = 35, training_path : str  = "data/input/training"):
        
        self.load_DEL_training_data(training_path)
        self.load_INV_training_data(training_path)
        self.img_size = self.DEL_xtrain.shape[1]
        self.n_labels = len(np.unique(self.DEL_ytrain))
        self.DELdetector = self.create_CNN(n_neurons)
        self.INVdetector = self.create_CNN(n_neurons)
        
    def load_DEL_training_data(self, training_path : str):
        
        """
        Loads npy files and split them into train and validaton set.

        Parameters
        ----------
        training_path : str
            Path to the npy files to load.

        """
        
        x_data = np.load(training_path + "/xnew_new.npy")
        y_data = np.load(training_path + "/ynew_new.npy")
        
        x_data = x_data[y_data != 1]
        y_data = y_data[y_data != 1]
        
        
        #x_data = x_data/5
        scaler = MinMaxScaler()
        x_data = scaler.fit_transform(x_data.reshape(-1, x_data.shape[-1])).reshape(x_data.shape)
	
        x_data = x_data.reshape((-1,x_data.shape[1], x_data.shape[2],1))
        
        y_data[y_data ==2]=1
        print(y_data.shape)
        print(y_data[y_data == 1].shape)
        
        self.DEL_xtrain, self.DEL_xvalid, self.DEL_ytrain, self.DEL_yvalid = train_test_split(
            x_data,y_data, train_size = 0.8)
           

            
    def load_INV_training_data(self, training_path : str):
        
        """
        Loads npy files and split them into train and validaton set.

        Parameters
        ----------
        training_path : str
            Path to the npy files to load.

        """

        x_data = np.load(training_path + "/xnew_new.npy")
        y_data = np.load(training_path + "/ynew_new.npy")
        
        x_data = x_data[y_data <= 1]
        y_data = y_data[y_data <= 1]
        
        
        #x_data = x_data/5
        scaler = MinMaxScaler()
        x_data = scaler.fit_transform(x_data.reshape(-1, x_data.shape[-1])).reshape(x_data.shape)
        
        x_data = x_data.reshape((-1,x_data.shape[1], x_data.shape[2],1))

        print(y_data.shape)
        print(y_data[y_data == 1].shape)
        
        self.INV_xtrain, self.INV_xvalid, self.INV_ytrain, self.INV_yvalid = train_test_split(
            x_data,y_data, train_size = 0.8)
                        
    def create_conv_block(self, x, filters : int = 32, kernel_size : int = 3, padding = "same", alpha : float = 0.1):

        x = Conv2D(filters , kernel_size, padding = padding, use_bias=False)(x)
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=alpha)(x)

        return x
    
    def create_residual_block(self, x, filters  : int = 32, kernel_size: int =3):

        input_x = x
        x = self.create_conv_block(x, filters, kernel_size)
        x = Conv2D(filters , kernel_size, padding="same", use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Add()([input_x, x])
        
        return x

    def create_CNN(self, n_neurons : int):
        
        """
        Builds model from scratch for training.

        Parameters
        ----------
        n_neurons : int
            Number of neurons in the two last layers.
        Returns
        -------
        keras.Model :
            Return the CNN model.
        """
        

        input_ = keras.layers.Input(name='input', shape=(128, 128, 1))
        
        x = self.create_residual_block(input_)
        x = self.create_residual_block(x)
        x = AveragePooling2D()(x)
        x = Flatten()(x)   
        
        x = Dense(n_neurons, activation="relu")(x)
        x = Dropout(0.2)(x)
        x = Dense(n_neurons, activation="relu")(x)
        x = Dropout(0.2)(x)
        output = Dense(self.n_labels, activation="softmax")(x)
    
        #Optimizer
        learning_rate = 1e-4 # Paramètres
        optimizer = Adam(learning_rate)
        
        model = keras.Model(input_, output)
        model.compile(
        optimizer= optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        )
        return model
        
    def create_RandomForest(self):
        return RandomForestClassifier(bootstrap = False)
            
    def train(self, n_epochs : int = 70):
        
        """
        Train model with training set.

        Parameters
        ----------
        n_epochs : int
            Number of epochs for the training.
        """
        """
        print("TRAIN DELdetector ON DATASET WITH {n_pic} PICTURES.".format(n_pic = len(self.DEL_ytrain)))
        # Stopper
        stopper = EarlyStopping(monitor="val_loss", patience=10, mode="min", verbose = 2,
                                restore_best_weights=False)
        # Reducelr
        reducelr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, verbose = 2, patience = 5)
        time_begin = time()
        self.history = self.DELdetector.fit(self.DEL_xtrain, self.DEL_ytrain, 
        validation_data = (self.DEL_xvalid, self.DEL_yvalid), verbose = True, 
        epochs = n_epochs, callbacks= [stopper, reducelr])
        time_end = time()
        
        time_tot = time_end - time_begin
        print("Training time is {hour_} h {min_} min {sec_} s.".format(hour_ = time_tot//3600, 
                        min_ = (time_tot%3600)//60, sec_ = (time_tot%3600)%60))
        
        print("Training score: {train_score}, Validation score: {val_score}".format(
        train_score = self.DELdetector.evaluate(self.DEL_xtrain,self.DEL_ytrain, verbose = 0)[1],
        val_score = self.DELdetector.evaluate(self.DEL_xvalid,self.DEL_yvalid, verbose = 0)[1]
        ))
    	"""
        print("TRAIN INVdetector ON DATASET WITH {n_pic} PICTURES.".format(n_pic = len(self.INV_ytrain)))

        print("TRAIN DELdetector ON DATASET WITH {n_pic} PICTURES.".format(n_pic = len(self.DEL_ytrain)))
        # Stopper
        stopper = EarlyStopping(monitor="val_loss", patience=10, mode="min", verbose = 2,
                                restore_best_weights=False)
        # Reducelr
        reducelr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, verbose = 2, patience = 5)
        time_begin = time()
        self.history = self.INVdetector.fit(self.INV_xtrain, self.INV_ytrain, 
        validation_data = (self.INV_xvalid, self.INV_yvalid), verbose = True, 
        epochs = n_epochs, callbacks= [stopper, reducelr])
        time_end = time()
        
        time_tot = time_end - time_begin
        print("Training time is {hour_} h {min_} min {sec_} s.".format(hour_ = time_tot//3600, 
                        min_ = (time_tot%3600)//60, sec_ = (time_tot%3600)%60))


        print("Training score: {train_score}, Validation score: {val_score}".format(
        train_score = self.DELdetector.evaluate(self.DEL_xtrain,self.DEL_ytrain, verbose = 0)[1],
        val_score = self.DELdetector.evaluate(self.DEL_xvalid,self.DEL_yvalid, verbose = 0)[1]
        ))
        #self.INVdetector.fit(self.INV_xtrain, self.INV_ytrain)
        #print("Training score:",self.INVdetector.score(self.INV_xtrain,self.INV_ytrain), "Validation score: ", 
        #self.INVdetector.score(self.INV_xvalid, self.INV_yvalid))
        
    def confusion_matrix(self):
        """
        Return the confusion matrices of each detector for the validation set.
        """
        return(confusion_matrix(self.DEL_yvalid, np.argmax(self.DELdetector.predict(self.DEL_xvalid), axis = 1)), 
        confusion_matrix(self.INV_yvalid, self.INVdetector.predict(self.INV_xvalid)))
    
    def plot(self):
        
        """
        Plot the evolution of loss, val_loss, accuracy, val_accuracy during the training.
        """
        
        # Plot training & validation accuracy values
        fig, ax = plt.subplots(1, 2)
        ax[0].plot(self.history.history["accuracy"], label="Train")
        ax[0].plot(self.history.history["val_accuracy"], label="Test")
        ax[0].set_title("Model accuracy")
        ax[0].set_ylabel("Accuracy")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylim(0, 1)
        # Plot training & validation loss values
        ax[1].plot(self.history.history["loss"], label="Train")
        ax[1].plot(self.history.history["val_loss"], label="Test")
        ax[1].set_title("Model loss")
        ax[1].set_ylabel("Loss")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylim(0, 2)
        plt.legend(loc="upper left")
        plt.show()
    
    
    
        
    def load_alea_scrambled(self, testing_path : str ="data/input/testing"):
        """
        Load a random scrambled matrix with breakpoints.tsv associated.
	
        Parameters:
        ----------
        testing_path : str 
            Path where all the matrices are stored.
        """
        n_run = 125
        ind_iter = np.random.randint(7,9)
        print("Iter: " + str(ind_iter))
        ind_run = np.random.randint(0,n_run)
        print("Run: " + str(ind_run))
    
        coord_SV = pd.read_csv(testing_path  + "/ITER" + str(ind_iter) + "/RUN_" + str(ind_run) + "/breakpoints.tsv", sep = "\t")
        
        truth = np.load(testing_path  + "/ITER" + str(ind_iter) + "/RUN_" + str(ind_run) + "/truth.npy")
        truth = np.log10(truth)
        return np.load(testing_path  + "/ITER" + str(ind_iter) + "/RUN_" + str(ind_run) + "/scrambled.npy"), coord_SV, truth
    
    def test(self, path : str = None, verbose : int = 1, thresold_INV : float = 6.5e-1, thresold_DEL : float = 5e-1):
        
        """
        Test model on new data and compute recall and precision scores.

        Parameters
        ----------
        path : str
            Path to the HiC matrix to test the model on it. If None, the model will be tested on a random Hi-C matrix.
        
        verbose : int
            Print or not the steps and the score of the test.
        
        thresold_INV : float
            Thresold for keeping INV during model.predict.
            
        thresold_DEL: float
            Thresold for keeping DEL during model.predict.
        """
        
        if path == None:
            scrambled, coord_SV, truth = self.load_alea_scrambled()
            plot = True
        else:
            scrambled = np.load(path + "/scrambled.npy")
            coord_SV = pd.read_csv(path + "/breakpoints.tsv", sep = "\t")
            plot = False
            
        scrambled = np.log10(scrambled)
        scrambled[scrambled == -inf] = 0
        scaler = MinMaxScaler()
        scrambled =  scaler.fit_transform(scrambled)
        
        coord_INV = coord_SV[coord_SV["sv_type"] == "INV"].sort_values("coord_start")
        coord_DEL = coord_SV[coord_SV["sv_type"] == "DEL"].sort_values("coord_start")
        
        list_ind_inv = list()
        list_ind_del = list()
        
        list_array_inv = list()
        list_array_del = list()
        
        for ind in range(0,len(coord_INV)):
        
            coord_start = coord_INV["coord_start"].values[ind]
            coord_end = coord_INV["coord_end"].values[ind]
            
            
        
            list_ind_inv += list(np.concatenate((np.array([coord_start- 1, coord_start, coord_start+1]),  
            np.array([coord_end- 1, coord_end, coord_end+1]))))
            list_array_inv.append(np.concatenate((np.array([coord_start- 1, coord_start, coord_start+1]),  
            np.array([coord_end- 1, coord_end, coord_end+1]))))
            
        for ind in range(0,len(coord_DEL)):
        
            coord = coord_DEL["coord_start"].values[ind]
   
            list_ind_del += [coord-3, coord-2,coord-1, coord, coord+1, coord+2, coord+3]
            list_array_del.append(np.array([coord-3, coord-2, coord-1, coord, coord+1, coord+2, coord+3]))
    
        array_good_indices = np.concatenate((np.array(list_ind_inv), 
                                        np.array(list_ind_del)))        
        array_good_indices = np.sort(array_good_indices)                                
 
        fen_zoom = self.img_size//2
        ind_beg = fen_zoom
        ind_end = len(scrambled) - fen_zoom
        
        number_INV_detected = 0
        number_DEL_detected = 0

        inds_INV_detected = list()
        inds_DEL_detected = list()
        
        if verbose == 1:
            
            print("--------------------------------------------------------------------") 
            print("BEGIN:", ind_beg, "END:", ind_end)
            print("THRESOLD INV:", thresold_INV,", THRESOLD DEL", thresold_DEL)

        for i in range(ind_beg, ind_end):
            
            slice_scrambled = scrambled[i-fen_zoom:i+fen_zoom,i-fen_zoom:i+fen_zoom]
            #slice_scrambled = self.DEL_scaler.transform(slice_scrambled)
            #scaler = MinMaxScaler()
            #slice_scrambled =  scaler.fit_transform(slice_scrambled)
            
            value_predicted_DEL = self.DELdetector.predict(slice_scrambled.reshape(1,
            slice_scrambled.shape[0],slice_scrambled.shape[1],1))[0,1]
            
            feature = np.matrix(scrambled[i,i-fen_zoom:i+fen_zoom])
            feature = feature.reshape(self.img_size)
            #scaler = MinMaxScaler()
            #slice_scrambled =  scaler.fit_transform(feature)
            
            
            value_predicted_INV = self.INVdetector.predict_proba(feature)[0,1]

                        
            if value_predicted_INV>=thresold_INV:
                
                #plt.matshow(slice_scrambled, cmap = "afmhot_r")
                #plt.title("INV line" + str(i))
        
                number_INV_detected += 1
                inds_INV_detected.append(i)
        
            if value_predicted_DEL >= thresold_DEL:
        	
                print(value_predicted_DEL)
        
                #plt.matshow(slice_scrambled, cmap = "afmhot_r")
                #plt.title("DEL line" + str(i))
        
                number_DEL_detected += 1
                inds_DEL_detected.append(i)
        if verbose == 1:
            print("TRUE INV COORD")
            print(coord_INV)
            print("COORD INV KEEPED")
            print(inds_INV_detected) 
            print("TRUE DEL COORD")
            print(coord_DEL) 
            print("COORD DEL KEEPED")
            print(inds_DEL_detected)   
            print("--------------------------------------------------------------------")  
            print("COMPUTE RECALL SCORES:")
        
        scores_detection_INV = list()
        for array in list_array_inv:            
            if np.max(array) >= ind_beg and np.min(array) <= ind_end:
                score_array = list()
                
                for k_indices in inds_INV_detected:        
                    score_array.append(len(np.where(array == k_indices)[0]) > 0)
                    
                if len(inds_INV_detected) > 0:
                    scores_detection_INV.append(max(score_array))
                    if scores_detection_INV[-1] == False and plot == True:
                        fen_zoom = 15
                        i = array[1]
                        slice_scrambled = scrambled[i-fen_zoom:i+fen_zoom,i-fen_zoom:i+fen_zoom]
                        #plt.matshow(slice_scrambled, cmap = "afmhot_r")
                        #plt.title("INV AT COORD " + str(i) + " NOT DETECTED") 
                        #plt.show()
                        
                        #plt.matshow(truth[i-fen_zoom:i+fen_zoom,i-fen_zoom:i+fen_zoom], cmap = "afmhot_r")
                        #plt.title("TRUTH FOR INV AT COORD " + str(i) + " NOT DETECTED") 
                        #plt.show()                        
                elif len(array) >0:
                    scores_detection_INV.append(0)
                
        score_detection_INV = np.mean(np.array(scores_detection_INV))
        
        scores_detection_DEL = list()
        for array in list_array_del:            
            if np.max(array) >= ind_beg and np.min(array) <= ind_end:
                score_array = list()
                for k_indices in inds_DEL_detected:        
                    score_array.append(len(np.where(array == k_indices)[0]) > 0)
                if len(inds_DEL_detected) > 0:
                    scores_detection_DEL.append(max(score_array))
                    if scores_detection_DEL[-1] == False and plot == True:
                        fen_zoom = 15
                        i = array[3]
                        slice_scrambled = scrambled[i-fen_zoom:i+fen_zoom,i-fen_zoom:i+fen_zoom]
                        #plt.matshow(slice_scrambled, cmap = "afmhot_r")
                        #plt.title("DEL AT COORD " + str(i) + " NOT DETECTED")
                        #plt.show() 
                        
                        #plt.matshow(truth[i-fen_zoom:i+fen_zoom,i-fen_zoom:i+fen_zoom], cmap = "afmhot_r")
                        #plt.title("TRUTH FOR DEL AT COORD " + str(i) + " NOT DETECTED") 
                        #plt.show()     
                                            
                elif len(array) >0:
                    scores_detection_DEL.append(0)
        score_detection_DEL = np.mean(np.array(scores_detection_DEL))
        
        if verbose == 1:
            print("The recall score for INV is {}.".format(score_detection_INV))         
            print("The recall score for DEL is {}.".format(score_detection_DEL)) 
                   
            print("--------------------------------------------------------------------")     
            print("COMPUTE PRECISION SCORES:")
        
        inds_keeped = np.concatenate((np.array(inds_INV_detected), np.array(inds_DEL_detected)))
        #print("LIST OF ALL INDEX KEEPED")
        #print(inds_keeped)      
        #print("LIST OF ALL TRUE INDICES")
        #print(array_good_indices)
        

        scores_precision_INV = []
        for k_indices in inds_INV_detected:
            scores_precision_INV.append(len(np.where(np.array(list_ind_inv) == k_indices)[0]) > 0)
            if scores_precision_INV[-1] == 0 and plot == True:
                fen_zoom = 15
                i = k_indices
                slice_scrambled = scrambled[i-fen_zoom:i+fen_zoom,i-fen_zoom:i+fen_zoom]
                #plt.matshow(slice_scrambled, cmap = "afmhot_r")
                #plt.title("FALSE POSITIVE (INV) AT COORD " + str(i)) 
                #plt.show()
            
        scores_precision_INV = np.array(scores_precision_INV)*1
        score_precision_INV = np.mean(scores_precision_INV)
        
        
        scores_precision_DEL = []
        for k_indices in inds_DEL_detected:
            scores_precision_DEL.append(len(np.where(np.array(list_ind_del) == k_indices)[0]) > 0)
            if scores_precision_DEL[-1] == False and plot == True:
                fen_zoom = 15
                i = k_indices
                slice_scrambled = scrambled[i-fen_zoom:i+fen_zoom,i-fen_zoom:i+fen_zoom]
                #plt.matshow(slice_scrambled, cmap = "afmhot_r")
                #plt.title("FALSE POSITIVE (DEL) AT COORD " + str(i))
                #plt.show()             
            
        scores_precision_DEL = np.array(scores_precision_DEL)*1
        score_precision_DEL = np.mean(scores_precision_DEL)
        
        if verbose == 1:
            print("The precision score for INV is {}.".format(score_precision_INV))
            print("The precision score for DEL is {}.".format(score_precision_DEL))        
        
        #score_precision = (score_precision_INV + score_precision_DEL)/2
        #weight_INV = (len(inds_INV_detected)/(len(inds_INV_detected) + len(inds_DEL_detected)))
        #weight_DEL = (len(inds_DEL_detected)/(len(inds_INV_detected) + len(inds_DEL_detected)))
        #score_precision = weight_INV*score_precision_INV + weight_DEL*score_precision_DEL
        
        #print("The mean precision score is {}.".format(score_precision))
        
        return score_detection_INV, score_detection_DEL, score_precision_INV, score_precision_DEL

    def predict(self,path, thresold_INV = 0.65, thresold_DEL = 0.5, verbose = 1):
        
        scrambled = np.load(path + "/scrambled.npy")
        scrambled = np.log10(scrambled)
        scrambled[scrambled == -inf] = 0
        scaler = MinMaxScaler()
        scrambled =  scaler.fit_transform(scrambled)
        #scrambled = scrambled/5

        fen_zoom = self.img_size//2
        ind_beg = fen_zoom
        ind_end = len(scrambled) - fen_zoom
        
        number_INV_detected = 0
        number_DEL_detected = 0

        inds_INV_detected = list()
        inds_DEL_detected = list()
        
        if verbose == 1:
            
            print("--------------------------------------------------------------------") 
            print("BEGIN:", ind_beg, "END:", ind_end)
            print("THRESOLD INV:", thresold_INV,", THRESOLD DEL", thresold_DEL)

        for i in range(ind_beg, ind_end):
            
            slice_scrambled = scrambled[i-fen_zoom:i+fen_zoom,i-fen_zoom:i+fen_zoom]
            #slice_scrambled = self.DEL_scaler.transform(slice_scrambled)
            #scaler = MinMaxScaler()
            #slice_scrambled =  scaler.fit_transform(slice_scrambled)
            
            value_predicted_DEL = self.DELdetector.predict(slice_scrambled.reshape(1,
            slice_scrambled.shape[0],slice_scrambled.shape[1],1))[0,1]
            
            feature = np.matrix(scrambled[i,i-fen_zoom:i+fen_zoom])
            feature = feature.reshape(self.img_size)
            #scaler = MinMaxScaler()
            #slice_scrambled =  scaler.fit_transform(feature)
            
            
            value_predicted_INV = self.INVdetector.predict_proba(feature)[0,1]

                        
            if value_predicted_INV>=thresold_INV:
                
                number_INV_detected += 1
                inds_INV_detected.append(i)
        
            if value_predicted_DEL >= thresold_DEL:
                print(i)
                print(value_predicted_DEL)
        
        
                number_DEL_detected += 1
                inds_DEL_detected.append(i)
        
        inds_DEL_detected_shift = np.concatenate((inds_DEL_detected[1:], np.zeros(1)))

        diff_array = inds_DEL_detected-inds_DEL_detected_shift
        list_zone_DEL = list() 

        ind_deb = 0
        for k in range(0, len(diff_array)):

            if abs(diff_array[k]) >2:

                list_zone_DEL.append(inds_DEL_detected[ind_deb:(k+1)])
                ind_deb = k+1

        for i in range(0, len(list_zone_DEL)):

            list_zone_DEL[i] = int(np.mean(list_zone_DEL[i]))

        print("INV DETECTED")
        print(inds_INV_detected)

        print("DEL DETECTED")
        print(list_zone_DEL)

        np.save("data/output/testing/index_detected/INV_index.npy", inds_INV_detected)
        np.save("data/output/testing/index_detected/DEL_index.npy", list_zone_DEL)    

        


    def save(self, model_dir : str = "data/models"):
        """
        Saves models configuration and weights to disk.
             
        Parameters
        ----------
        model_dir : str
        Directory where the model will be saved.
        """
        #model_json = self.DELdetector.to_json()
        #with open(join(model_dir + "/DEL/", "model.json"), "w") as json_file:
        #    json_file.write(model_json)
        #self.DELdetector.save_weights(join(model_dir+ "/DEL/", "weights.h5"))

        model_json = self.INVdetector.to_json()
        with open(join(model_dir + "/INV/", "model.json"), "w") as json_file:
            json_file.write(model_json)
        self.INVdetector.save_weights(join(model_dir+ "/INV/", "weights.h5"))
       

    def load(self, model_dir="data/models"):
        """
        Loads a trained neural network from a json file and a 
        RandomForestClassifier.
            
        Parameters
        ----------
        model_dir : str
        Directory where the model will be loaded.
        """
        with open(join(model_dir + "/DEL/", "model.json"), "r") as json_file:
            loaded_model_json = json_file.read()
        loaded_model = model_from_json(loaded_model_json)
        loaded_model.load_weights(join(model_dir + "/DEL/", "weights.h5"))
        loaded_model.compile(
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        optimizer="adam",
        )
        self.DELdetector = loaded_model
        self.INVdetector = joblib.load(model_dir + "/INV/random_forest.joblib")

