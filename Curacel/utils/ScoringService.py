import pickle
import numpy as np

class ScoringService(object):
    model = None                # Where we keep the model when it's loaded
    preprocessor = None         # Where we keep the preprocessor when it's loaded
    @classmethod
    def get_model(cls,model_path):
        """Get the model object for this instance, loading it if it's not already loaded."""
        if cls.model == None:
            # load the model from disk
            with open(model_path, 'rb') as file:
                cls.model = pickle.load(file)
        return cls.model


    @classmethod
    def predict(cls, data, model_path):
        clf = cls.get_model(model_path)
        predictions = clf.predict(data)
        predictions = np.exp(predictions)
        return predictions
    
    @classmethod
    def get_preprocessor(cls,preprocessor_path):
        """Get the preprocessor object for this instance, loading it if it's not already loaded."""
        if cls.preprocessor == None:
            with open(preprocessor_path, 'rb') as file:
                cls.preprocessor = pickle.load(file)
        return cls.preprocessor