'''
This is a helper file for choosing which model to create.
'''
import tensorflow as tf

def create_model(args, input_shape):
    # If using CPU or single GPU
    if args.gpus <= 1:
        if args.net == 'udet':
            from UDet import UDet
            model = UDet(input_shape)
            return [model]
        if args.net == 'bifpn':
            from Encoder_BIFPN import BIFPN
            model = BIFPN(input_shape)
            return [model]
        if args.net == 'unet':
            from unet import UNet
            model = UNet(input_shape)
            return [model]
        else:
            raise Exception('Unknown network type specified: {}'.format(args.net))
    # If using multiple GPUs
    else:
        with tf.device("/cpu:0"):
            if args.net == 'udet':
                from BIFPN import UDet
                model = UDet(input_shape)
                return [model]
            if args.net == 'unet':
                from unet import UNet
                model = UNet(input_shape)
                return [model]
            else:
                raise Exception('Unknown network type specified: {}'.format(args.net))
