import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import shutil



class PredictionPipeline:
    def __init__(self,filename):
        self.filename =filename


    
    def predict(self):
        ## load model
        # model = load_model(os.path.join("artifacts","training", "model.h5"))

        training_model_path = os.path.join("artifacts", "training", "model.h5")
        target_model_folder = "model"
        target_model_path = os.path.join(target_model_folder, "model.h5")
        
        # Create the target directory if it doesn't exist
        if not os.path.exists(target_model_folder):
            os.makedirs(target_model_folder)

        # Copy the model file
        shutil.copy(training_model_path, target_model_path)

        model = load_model(target_model_path)

        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (224,224))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = np.argmax(model.predict(test_image), axis=1)
        print(result)

        if result[0] == 1:
            prediction = 'Normal'
            return [{ "image" : prediction}]
        else:
            prediction = 'Adenocarcinoma Cancer'
            return [{ "image" : prediction}]
