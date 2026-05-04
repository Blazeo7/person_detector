from abc import ABC, abstractmethod
import numpy as np

class BaseDetector(ABC):
    """
    Enforces a standard API for training and evaluation.
    """
    
    @abstractmethod
    def fit(self, X: list, y: np.ndarray):
        """
        Train the model using the provided features and labels.
        
        Args:
            X: A list of feature arrays. For audio, this is a list of 2D arrays 
               (frames x features). For images, a list of 1D flattened arrays.
            y: A 1D array of binary labels (1 for TARGET, 0 for NON-TARGET).
               
        Returns:
            self
        """
        pass

    @abstractmethod
    def predict_proba(self, X: list) -> np.ndarray:
        """
        Compute the continuous confidence score for each sample.
        Higher score = more confident that the sample belongs to the target person.
        
        Args:
            X: A list of feature arrays.
            
        Returns:
            1D numpy array of float scores.
        """
        pass

    def predict(self, X: list) -> np.ndarray:
        """
        Make hard decisions (1 or 0) based on the predicted scores.
        The assignment requires an apriori probability of 0.5, which typically
        means the decision threshold is strictly at 0.0.
        """
        scores = self.predict_proba(X)
        return (scores > 0).astype(int)