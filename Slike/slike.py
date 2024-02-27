from models import *

class slike : 

    models = {
        'clova' : {
            'craft' : [CRAFT()]
        }
    }

    def get_model(self , model , return_tensors = 'pt') : 

        if model : 

            model = model.split('/')

            if return_tensors == 'pt' : return self.models[model[0]][model[1]][0]

            # TODO : Add some edge conditions 
            # TODO : Add Tensorflow Model
            # TODO : Add hugging Face Model 