# -*- coding: utf-8 -*-
"""
Created on Sat Aug 17 12:26:27 2019

@author: Abdelhamid bouzid
"""

import torch

class GuidedBackprop():
    """
       Produces gradients generated with guided back propagation from the given image
    """
    def __init__(self, model,):
        self.model = model
        self.gradients = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        # Register hook to the first layer
        first_layer = list(self.model._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
            Updates relu activation functions so that
                1- stores output in forward pass
                2- imputes zero for gradient values that are less than zero
        """
        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(grad_in[0], min=0.0)
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for pos, module in self.model._modules.items():
            for pos_1, residual in  module._modules.items():
                for pos_2, module_2 in  residual._modules.items():
                    for pos_3, module_3 in  module_2._modules.items():
                        if isinstance(module_3, torch.nn.modules.activation.LeakyReLU):
                            #print(module_3)
                            module_3.register_backward_hook(relu_backward_hook_function)
                            module_3.register_forward_hook(relu_forward_hook_function)

    def generate_gradients(self, input_image,unite,block,cnn_layer, filter_pos):
        
        # Forward pass
        flage      = 0
        x          = input_image
        for pos, module in self.model._modules.items():
            print(pos)

            if pos == unite:
                for pos_1, residual in  module._modules.items():
                    if pos_1==block:
                        for pos_2, module_2 in  residual._modules.items():
                            if pos_2=="pre_act":
                                x  = module_2(x)
                            elif pos_2=="identity":
                                x_1 = module_2(x)
                            else:
                                for pos_3, module_3 in  module_2._modules.items():
                                    print(pos_3)
                                    x  = module_3(x)
                                    if pos_3==cnn_layer:
                                        flage=1
                                        break

                            if flage==1:
                                break
                    else:
                        x  = residual(x)
                    if flage==1:
                        break
            else:
                x = module(x)
                print(x.shape)

            if flage==1:
                break

            print("###################################################")
                
        # Backward pass
        self.model.zero_grad()
        conv_output = torch.sum(torch.abs(x[0, filter_pos]))
        conv_output.backward()
        
        #convert to numpy
        gradients   = self.gradients.cpu()
        gradients_as_arr = gradients.data.numpy()[0]
        return gradients_as_arr