import torch
import torch.nn as nn


def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Identity):
            pass

        elif isinstance(module, nn.Conv2d):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)

        elif isinstance(module, nn.ConvTranspose2d):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)

        elif isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            torch.nn.init.constant_(module.weight, 1)
            torch.nn.init.constant_(module.bias, 0)

        elif isinstance(module, nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)

        elif isinstance(module, nn.LSTM) or isinstance(module, nn.LSTMCell):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.kaiming_normal_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
