from torch import nn


class CnnInit(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_layers):
        super(CnnInit, self).__init__()
        """
        :param in_channels: number of initial layers
        :param out_channels: number of final layers
        :param hidden_layers: array with dimension matching the number of hidden layers, with the 
                              dimensions of the layers.
        """
        super().__init__()

        layers = [nn.Conv2d(in_channels, hidden_layers[0], kernel_size=(3, 3), padding=1),
                  nn.ReLU()]
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Conv2d(hidden_layers[i - 1], hidden_layers[i], kernel_size=(3, 3), padding=1))
            layers.append(nn.ReLU())
        layers.append(nn.Conv2d(hidden_layers[-1], out_channels, kernel_size=(3, 3), padding=1))
        layers.append(nn.ReLU())

        self.cnn = nn.Sequential(*layers)

    def forward(self, x):
        return self.cnn(x)


class CnnShrink(nn.Module):
    """
    Cnn that shrinks by factor 8 and keeps the channel number the same.
    """
    def __init__(self, in_channels):
        super(CnnShrink, self).__init__()

        # EACH layer shrinks transverse size by a factor of 2.
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=(4, 4), stride=(2, 2), padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        return self.cnn(x)
