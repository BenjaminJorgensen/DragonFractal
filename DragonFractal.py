import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as mstyle
import matplotlib.animation as anime #lol
import catppuccin

# Setting up colours and environment settings
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mstyle.use(catppuccin.PALETTE.macchiato.identifier)

# Verify if imports succeeded
print(torch.__version__)
print(device)
