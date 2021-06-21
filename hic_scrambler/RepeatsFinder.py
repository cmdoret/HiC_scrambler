import numpy as np
import complexity_function as cf

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

class RepeatsFinder(object):
    """
    Handles to detect repeats with the help of Lempel-Ziv complexity. Will create an array which 
    correspond to different positions. For each position, RepeatsFinder take a sequence in a certain 
    window near the position and compute the complexity for this sequence.

    The feature is the minimum of the array.

    Examples
    --------
        RFinder = RepeatsFinder()
        RFinder.predict(coord, path)


    Attributes
    ----------
    size_win : int
        Size of the window where the complexity of the sequence will be computed for one position.

    size_tab : int
        Size of the array which store the complexity for each position.

    chrom : str
        Name of the chroms where modifications have been applied.
    """

    def __init__(self, size_win : int = 40, size_tab : int = 30, chrom : str = "Sc_chr04"):
	
        self.load_data()
        self.create_and_train_model()
        self.size_win = size_win
        self.size_tab = size_tab
        self.chrom = chrom


    def load_data(self, training_path : str = "data/input/training_repeats"):
        """
        Load data to train model.
        
        Parameters
        ----------
        training_path : str
            Path where the training set where there is example of repeats.
        """
        array_repeats = np.load(training_path + "/complexity_repeats.npy")
        array_SV = np.load(training_path + "/complexity_SV.npy")
        
        labels_repeats = np.ones(len(array_repeats))
        labels_SV = np.zeros(len(array_SV))
        

        features = np.concatenate((array_repeats, array_SV)).reshape((-1,1))
        labels = np.concatenate((labels_repeats, labels_SV))

        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(features, labels)
        



    def create_and_train_model(self):
        """
        Create DecisionTreeClassifier to detect repeats.
        """
        self.Classifier = DecisionTreeClassifier(max_depth = 1).fit(self.X_train, self.y_train)

    
    def predict(self, coord : int, path : str , chrom_name : str, verbose : bool = True):
	    """
	    Predict if there is a repeat or not.

        Parameters
        ----------
        coord : int
            Coordinate that you want to test if it is a repeat or not.

        path : str
            Path where the genome file is.

        chrom_name : str
            Chromosome of the genome you want to test.

        verbose : bool
            To remove or not verbose.

        Returns
        -------
        bool :
            Returns a boolean. True if it is a repeat, False it is not.
	    """
        
            complexity = np.zeros(2*self.size_tab + 1)
            
            for k in range(-self.size_tab,self.size_tab+1):
                ind_beg = coord + k - self.size_win//2
                ind_end = coord + k + self.size_win//2
                seq = cf.load_seq(path, chrom_id = chrom_name, ind_beg = ind_beg, ind_end = ind_end)
                complexity[k] = cf.lempel_complexity(seq)
            
            
            min_complexity = np.min(complexity)

            label_predicted = self.Classifier.predict(np.array([min_complexity]).reshape(-1,1))[0]

            if label_predicted ==1:
                if verbose == True:
                    print("Repeat")
                return True
            else:
                if verbose == True:
                    print("Not a repeat")
                return False
