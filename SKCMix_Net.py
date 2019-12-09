import torch.nn as nn
from collections import OrderedDict
from layers import SKMix_Block, _Transition, _DeTransition
class SKMix_net(nn.Module):
    def __init__(self, block_encoder_config=(2, 2, 2), block_decoder_config=(2, 2, 2), num_init_features=64,
                 expansion=2, k1=32, k2=16, drop_rate=0,
                 ):
        super(SKMix_net, self).__init__()


        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=1, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
        ]))
        # Encoder
        num_features = num_init_features
        #print(num_features)
        for i, num_layers in enumerate(block_encoder_config):
            trans = _Transition(num_input_features=num_features, num_output_features=num_features * 2)
            self.features.add_module('transition%d' % (i + 1), trans)
            num_features = max(num_features * 2, k1)
            block = SKMix_Block(num_layers=num_layers, in_channels=num_features, expansion=expansion, k1=k1, k2=k2,
                                drop_rate=drop_rate)
            self.features.add_module('eblock%d' % (i + 1), block)
            num_features = num_features + num_layers * k2
        #print(num_features)

        # Decoder
        for i, num_layers in enumerate(block_decoder_config):
            print(num_features)
            block = SKMix_Block(num_layers=num_layers, in_channels=num_features, expansion=expansion, k1=k1, k2=k2,
                                drop_rate=drop_rate)
            self.features.add_module('block%d' % (i + 1), block)
            num_features = num_features + num_layers * k2
            detrans = _DeTransition(num_features,num_features // 2)
            self.features.add_module('detransition%d' %(i + 1),detrans)
            num_features = max(num_features // 2, k1)
        print(num_features)
        # Final
        self.features.add_module('conv_last', nn.Sequential(OrderedDict([

            ('conv_last_1', nn.Conv2d(num_features, 64, kernel_size=1, stride=1, padding=0, bias=False)),
            ('conv_last_2', nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0, bias=False)),

        ])))

    def forward(self, x):
        out = self.features(x)
        #print(out.size())
        return out

print(SKMix_net())