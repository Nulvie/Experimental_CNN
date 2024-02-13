import torch
from torch import nn
    
class ResidualBlock(nn.Module):
    def __init__(self, out_channels, kernel_size):
        super().__init__()

        self.conv_stack = nn.Sequential(
            nn.LazyConv1d(out_channels, kernel_size, padding='same'),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(0.1),

            nn.Conv1d(out_channels, out_channels, kernel_size, padding='same'),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(),
            nn.Dropout(0.1)
        )

        self.shortcut = nn.LazyConv1d(out_channels, 1)

    def forward(self, x):
        return self.conv_stack(x) + self.shortcut(x)

class combined_CNN(nn.Module):
    def __init__(self, include_dotb=True):
        super().__init__()
        self.include_dotb = include_dotb

        self.setup_seqs_conv_block()
        self.setup_dotb_conv_block()
        self.setup_fc_block()

        self.loss_fn = nn.MSELoss()
        # if include_dotb:
        #     self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=0)
        # else:
        #     params = list(self.seqs_conv_stack.parameters()) + list(self.first_fc_x.parameters()) + list(self.fc_stack.parameters())
        #     self.optimizer = torch.optim.Adam(params, lr=0.01, weight_decay=0)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=0)


    def setup_conv_block(self):
        num_filters = 64
        res_stack = nn.Sequential(
            ResidualBlock(num_filters, 3),
            ResidualBlock(num_filters, 3),
            ResidualBlock(num_filters, 3),
            ResidualBlock(num_filters, 3),
            ResidualBlock(num_filters, 3)
        )
        return res_stack

    def setup_seqs_conv_block(self):
        conv_stack = self.setup_conv_block()
        self.seqs_conv_stack = nn.Sequential(*conv_stack)

    def setup_dotb_conv_block(self):
        conv_stack = self.setup_conv_block()
        self.dotb_conv_stack = nn.Sequential(*conv_stack)


    def setup_fc_block(self):
        # setup fully connected layers
        fc_stack = [
            nn.LazyLinear(64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),

            nn.LazyLinear(16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5)
        ]

        first_fc_x = [
            nn.LazyLinear(64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
        ]
        first_fc_z = [
            nn.LazyLinear(64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
        ]

        # add final layer to one neuron
        fc_stack.append(nn.LazyLinear(1))
        self.fc_stack = nn.Sequential(*fc_stack)
        self.first_fc_x = nn.Sequential(*first_fc_x)
        self.first_fc_z = nn.Sequential(*first_fc_z)

    def forward(self, x, z):
        seqs = x

        # separate conv layers for x and z
        x = self.seqs_conv_stack(x)
        x = nn.Flatten()(x)

        # separate FC layers for x and z
        x = self.first_fc_x(x)

        # Combine x and z
        combined = 0
        if self.include_dotb: 
            z = self.dotb_conv_stack(z)
            z = nn.Flatten()(z)

            z = self.first_fc_z(z)
            
            combined = self.fc_stack(x + z)
        # Run the conv and fc on seqs again
        else: 
            z = self.dotb_conv_stack(seqs)
            z = nn.Flatten()(z)

            z = self.first_fc_z(z)

            combined = self.fc_stack(x + z)
        return combined

    def __str__(self) -> str:
        return super().__str__()