# Run with pytorch 1.7.1 and torchvision 0.8.2

import os
import torch
from clip.clip import _download, _MODELS

model_filepath = _download(_MODELS['RN50'], os.path.expanduser("~/.cache/clip"))

model = torch.jit.load(model_filepath, 'cpu')
torch.save(model.state_dict(), 'RN50_1.5.pt')
