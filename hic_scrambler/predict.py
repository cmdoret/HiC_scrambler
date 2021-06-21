# Script used to predict SV on a random scrambled matrix. 

from hic_scrambler.models.SVDetector import SVDetector
import numpy as np
import pandas as pd

path = "data/input/testing/"

def test_matrix(path_ = None):
    Detector = SVDetector()
    Detector.load()
    
    return Detector.predict(path_, verbose = 1)

def test_score(Detector, inv_thresold, del_thresold):
    
    
    scores_inv_det = list()
    scores_del_det = list()
    scores_inv_prec = list()
    scores_del_prec = list()
    
    
    print("THRESOLD INV:", inv_thresold,", THRESOLD DEL", del_thresold)
        
    # Test for all images
    for ind_iter in range(7,9):
        for ind_run in range(0,125):
            
            scores = Detector.test(path + "ITER" + str(ind_iter) + "/RUN_" + str(ind_run), verbose = 0, thresold_INV = inv_thresold, thresold_DEL = del_thresold)
            
            if np.isnan(scores[0]) == False:
            	scores_inv_det.append(scores[0])
            	
            if np.isnan(scores[1]) == False:
            	scores_del_det.append(scores[1])
            	
            if np.isnan(scores[2]) == False:
            	scores_inv_prec.append(scores[2])                    	            

            if np.isnan(scores[3]) == False:
            	scores_del_prec.append(scores[3])  
            
            if ind_run%20 == 0:
                print("AT ITER {iter_}, RUN {run_}:".format(iter_ = ind_iter, 
                    run_ = ind_run))
                
                
                print("The test recall score for INV is {}.".format(np.mean(np.array(scores_inv_det)))) 
                print("The test recall score for DEL is {}.".format(np.mean(np.array(scores_del_det))))                 
                print("The test precision score for INV is {}.".format(np.mean(np.array(scores_inv_prec))))
                print("The test precision score for DEL is {}.".format(np.mean(np.array(scores_del_prec))))          
    
    # Mean of all scores
    score_inv_det = np.mean(np.array(scores_inv_det))
    score_del_det = np.mean(np.array(scores_inv_det))
    score_inv_prec = np.mean(np.array(scores_inv_det))
    score_del_prec = np.mean(np.array(scores_inv_det))
    
    print("----------------------------------------------------------------------")
    print("FINAL SCORES ARE:")
    print("The test recall score for INV is {}.".format(score_inv_det)) 
    print("The test recall score for DEL is {}.".format(score_del_det))                 
    print("The test precision score for INV is {}.".format(score_inv_prec))
    print("The test precision score for DEL is {}.".format(score_del_prec))      
    
    df = pd.DataFrame()
    df["Score INV detection"] = [score_inv_det]
    df["Score DEL detection"] = [score_del_det]
    df[ "Score INV precision"] = [score_inv_prec]
    df["Score DEL precision"] = [score_del_prec]

    # Save
    #df.to_csv("data/output/testing/score" +  str(100*inv_thresold) +  ".csv")
    return np.array(scores_inv_det), np.array(scores_del_det), np.array(scores_inv_prec), np.array(scores_del_prec)
     
if __name__ == "__main__":
    
    print(test_matrix(path_ = "data/input/testing/ITER7/RUN_75"))
    
           
