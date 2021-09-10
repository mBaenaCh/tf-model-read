from keras.models import model_from_json
import numpy as np

class FlowerRecognitionModel(object):

    FLOWERS_CLASSES = ['daisy', 'dandelion', 'rose', 
                       'sunflower', 'tulip']
    
    def __init__(self, model_json_file, model_weights_file):
        with open(model_json_file, "r") as json.file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model._make_predict_function()
    
    def predict_img(self, img):
        self.preds = self.loaded_model.predict(img)
        return FlowerRecognitionModel.FLOWERS_CLASSES[np.argmax(self.preds)]