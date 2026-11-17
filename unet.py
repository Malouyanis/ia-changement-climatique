"""Architecture UNet (Ronneberger et al., 2015), partagée par les notebooks 03 et 04."""
import torch
import torch.nn as nn


class DoubleConv(nn.Sequential):
    """Deux convolutions 3x3, chacune suivie d'une normalisation de batch et d'une ReLU."""

    def __init__(self, c_in, c_out):
        super().__init__(
            nn.Conv2d(c_in, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
            nn.BatchNorm2d(c_out),
            nn.ReLU(inplace=True),
        )


class UNet(nn.Module):
    """UNet à 4 niveaux.

    in_ch     : nombre de bandes en entrée (6 ici)
    n_classes : nombre de classes en sortie
    base      : nombre de filtres au premier niveau (doublé à chaque niveau)
    """

    def __init__(self, in_ch, n_classes, base=32, dropout=0.1):
        super().__init__()
        c = [base, base * 2, base * 4, base * 8, base * 16]

        # Encodeur (descente) : « qu'est-ce qu'il y a ? »
        self.enc = nn.ModuleList([
            DoubleConv(in_ch, c[0]),
            DoubleConv(c[0], c[1]),
            DoubleConv(c[1], c[2]),
            DoubleConv(c[2], c[3]),
        ])
        self.pool = nn.MaxPool2d(2)

        # Fond du U
        self.bottom = nn.Sequential(DoubleConv(c[3], c[4]), nn.Dropout2d(dropout))

        # Décodeur (remontée) : « où exactement ? »
        self.up = nn.ModuleList([
            nn.ConvTranspose2d(c[4], c[3], 2, stride=2),
            nn.ConvTranspose2d(c[3], c[2], 2, stride=2),
            nn.ConvTranspose2d(c[2], c[1], 2, stride=2),
            nn.ConvTranspose2d(c[1], c[0], 2, stride=2),
        ])
        self.dec = nn.ModuleList([
            DoubleConv(c[4], c[3]),
            DoubleConv(c[3], c[2]),
            DoubleConv(c[2], c[1]),
            DoubleConv(c[1], c[0]),
        ])

        # Une valeur par classe et par pixel
        self.head = nn.Conv2d(c[0], n_classes, 1)

    def forward(self, x):
        skips = []
        for bloc in self.enc:
            x = bloc(x)
            skips.append(x)          # copie gardée pour la connexion de saut
            x = self.pool(x)
        x = self.bottom(x)
        for up, dec, skip in zip(self.up, self.dec, reversed(skips)):
            x = up(x)
            x = torch.cat([skip, x], dim=1)   # connexion de saut
            x = dec(x)
        return self.head(x)