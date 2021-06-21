import os # To remove warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import pandas as pd
import numpy as np
from os.path import join
from keras.models import model_from_json
import joblib



class Optimizer(object):

    def __init__(self, img_size = 128):
    	

        self.load()
        self.img_size = 128
        self.optimise()

    def optimise(self, optim_path : str = "data/input/optim/"):
        
        thresolds_INV = np.arange(0,1,0.4)
        thresolds_DEL = np.arange(0,1,0.4)
        scores = np.zeros((thresolds_INV.shape[0], thresolds_DEL.shape[0]))
        
        
        for i in range(0, len(thresolds_INV)):
            for j in range(0, len(thresolds_DEL)):
                
                print("Test for inv_thresold = {inv}, del_thresold = {del_}".format(inv = thresolds_INV[i], del_ = thresolds_INV[j]))  
                scores[i,j]=self.score_test_all_images(thresolds_INV[i], thresolds_DEL[j])
        
        self.best_INV_thresold, self.best_DEL_thresold =  np.where(scores == np.max(scores)) # np.argmax doesn't give row and column.
         
        # If several maximums
        self.best_INV_thresold = self.best_INV_thresold[0]
        self.best_DEL_thresold = self.best_DEL_thresold[0]
         
        print("BEST INV THRESOLD: {inv}, BEST DEL THRESOLD: {del} WITH SCORE OF {score}.".format(inv = self.best_INV_thresold, 
	    del_ = self.best_DEL_thresold, score = scores[self.best_INV_thresold, self.best_DEL_thresold]))	

    def load(self, model_dir="data/models"):
        """
        Loads a trained neural network from a json file
            
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

    def save(self,optim_path : str = "data/output/optim/"):
        np.save(optim_path + "optim.npy", np.array([self.best_INV_thresold, self.best_DEL_thresold]))
	 
    def find_indices(self, scrambled, thresold_INV, thresold_DEL):
    
        fen_zoom = self.img_size//2
        ind_beg = fen_zoom
        ind_end = len(scrambled) - fen_zoom

        indices_keeped = list()

        for i in range(ind_beg, ind_end):
            
            slice_scrambled = scrambled[i-fen_zoom:i+fen_zoom,i-fen_zoom:i+fen_zoom]
            
            value_predicted_DEL = self.DELdetector.predict(slice_scrambled.reshape(1,
            slice_scrambled.shape[0],slice_scrambled.shape[1],1))[0,1]
            
            feature = np.matrix(scrambled[i,i-fen_zoom:i+fen_zoom])
            feature = feature.reshape(self.img_size)
            value_predicted_INV = self.INVdetector.predict_proba(feature)[0,1]

                        
            if value_predicted_INV>=thresold_INV:
                
               
                indices_keeped.append(i)
        
            if value_predicted_DEL >= thresold_DEL:

                indices_keeped.append(i)
        
        return indices_keeped
        
    def score_indices(self, keep_indices, SV_dataframe):
    
        fen_zoom = self.img_size//2
        #ind_beg = fen_zoom
        #ind_end = len(scrambled) - fen_zoom
    
        inv_dataframe = SV_dataframe[SV_dataframe["sv_type"]=="INV"]
        del_dataframe = SV_dataframe[SV_dataframe["sv_type"]=="DEL"]
    
        list_ind_inv = list()
        list_del_inv = list()
    
        for ind in range(0,len(inv_dataframe)):
        
            coord_start = inv_dataframe["coord_start"].values[ind]
            coord_end = inv_dataframe["coord_end"].values[ind]
        
        
            list_ind_inv += list(np.arange(coord_start-1,coord_end+2))
        
        for ind in range(0,len(del_dataframe)):
        
            coord = del_dataframe["coord_start"].values[ind]
   
            list_del_inv += [coord-1, coord, coord+1]
    
        array_good_indices = np.concatenate((np.array(list_ind_inv), 
                                        np.array(list_del_inv)))
    
        scores = []
        for k_indices in keep_indices:
    	    scores.append(len(np.where(array_good_indices == k_indices)[0]) > 0)
        scores = np.array(scores)*1
        score = np.mean(scores)
    
        return score    



    
    def score_test_all_images(self, thresold_INV, thresold_DEL, optim_path : str = "data/input/optim/"):
    
        inds_iter = [7,8]
        inds_run = np.arange(0,125)
    
        scores = []   
        for ind_iter in inds_iter:
            for ind_run in inds_run:
                
                scrambled = np.load(optim_path + "ITER" + str(ind_iter) + "/RUN_" + str(ind_run) + "/scrambled.npy")
                breakpoints = pd.read_csv(optim_path + "ITER" + str(ind_iter) + "/RUN_" + str(ind_run) + "/breakpoints.tsv", sep = "\t")
                scores.append(self.score_indices(self.find_indices(scrambled, thresold_INV, thresold_DEL), 
                        breakpoints))
                
                if ind_run%10 == 0:
                    print("ITER {iter_}, RUN {run}.".format(iter_ = ind_iter, run = ind_run))
    
        scores = np.array(scores)
        return np.mean(scores)