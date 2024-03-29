import os
import torch
import torch.nn as nn
from torchvision import models
class pred():
    def __init__(self):
        pass
    def predi(self,img):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # create model
        model = models.resnet34(pretrained=False)
        in_channel = model.fc.in_features
        model.fc = nn.Linear(in_channel, 5)
        model.to(device)

        # load model weights
        weights_path = "./checkpoints/CL_resNet34.pth"
        assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
        model.load_state_dict(torch.load(weights_path, map_location=device))

        # prediction
        model.eval()
        with torch.no_grad():
            output_s = model(img.to(device))
            predict_y = torch.max(output_s, dim=1)[1].item()
        return int(predict_y)


