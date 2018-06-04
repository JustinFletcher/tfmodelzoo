# Import all models.
import lenet
import alexnet
import resnet


# function to print the tensor shape.  useful for debugging
def print_tensor_shape(tensor, string):
    '''
    input: tensor and string to describe it
    '''

    if __debug__:
        print('DEBUG ' + string, tensor.get_shape())


class TensorFlowModelZoo(object):

    def get_model(self, model_name, model_params=[]):

        if model_name == 'lenet':

            tfmodel = lenet.LeNetTensorFlowModel()

            return(tfmodel)

        if model_name == 'alexnet':

            tfmodel = alexnet.AlexNetTensorFlowModel()

            return(tfmodel)

        if model_name == 'vgg-16':

            print(model_name + " is not yet implemented.")
            raise NotImplementedError

        if model_name == 'googlenet':

            print(model_name + " is not yet implemented.")
            raise NotImplementedError

        if model_name == 'resnet-152':

            print(model_name + " is not yet implemented.")
            raise NotImplementedError

        if model_name == 'resnet50':

            tfmodel = resnet.ResNet50TensorFlowModel()

            return(tfmodel)

        else:

            print(model_name + " is not a recognized model name.")
            raise NotImplementedError

    # TODO: Build a model registration inferface and call it for other models.
