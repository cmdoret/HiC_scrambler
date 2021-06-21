# Script used to create and train model to predict SV. 

from hic_scrambler.models.SVDetector import SVDetector

if __name__ == "__main__":
    
    Detector = SVDetector()
    Detector.train()
    #Detector.plot()
    
    conf_mat = Detector.confusion_matrix()
    print("Confusion matrix for DELdetector:")
    print(conf_mat[0])
    print("Confusion matrix for INVdetector:")
    print(conf_mat[1])        
    
    Detector.save()
