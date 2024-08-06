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
def generate_new_points(points: torch.Tensor) -> torch.Tensor:

    # Generate midpoints by effectively doubling the input points
    midpoints = (points[:-1] + points[1:]) / 2

    # Split the midpoints based on the direction they're about to extend to
    dx = points[1:, 0] - points[:-1, 0]
    dy = points[1:, 1] - points[:-1, 1]

    # Prepare directions tensor
    directions = torch.empty_like(midpoints)
    directions[::2] = torch.stack([-dy[::2], dx[::2]], dim=1) / 2
    directions[1::2] = torch.stack([dy[1::2], -dx[1::2]], dim=1) / 2

    # Calculate new midpoints
    new_midpoints = midpoints + directions

    # Interleave the original points and new midpoints
    interleaved_points = torch.zeros((len(points) + len(new_midpoints), 2), device=device)
    interleaved_points[0::2] = points
    interleaved_points[1::2][:len(new_midpoints)] = new_midpoints
    interleaved_points[-1] = points[-1]
    return interleaved_points
