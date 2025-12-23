import torch
from torch import nn
import torchvision.transforms as transforms
from torchvision.models import vgg16

from learning.dataset import YOLODataset


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv2d(3, 20, 5)
        # self.conv2 = nn.Conv2d(20, 20, 5)
        # self.seq = nn.Sequential(
        #     # nn.Conv2d(3, 20, 5),
        #     # nn.Conv2d(20, 20, 5)
        #     # nn.MaxPool2d(2)
        #     nn.AdaptiveAvgPool2d((256,256))
        # )
        self.feature_extractor = vgg16().features
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 14 * 14, 4096),
            nn.ReLU(),
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 8)
        )

    def forward(self, x):
        # x = torch.nn.functional.relu(self.conv1(x))
        # return torch.nn.functional.relu(self.conv2(x))
        # return self.seq(x)
        x = self.feature_extractor(x)
        return self.fc_layers(x)


if __name__ == '__main__':
    model = Model()
    # dataset = YOLODataset("F:\Python\Project\psychic-online-ai\\test-data\\yolo\images",
    #                       "F:\Python\Project\psychic-online-ai\\test-data\\yolo\labels",
    #                       transforms.Compose([transforms.ToTensor(), transforms.Resize((512, 512))]), None)
    # image, target = dataset[0]
    # output = model(image)
    # torch.onnx.export(model, image, "tudui.onnx")
    # print(output.shape)
    print(model)
    input = torch.rand(64, 3, 448, 448)
    output = model(input)
    print(output)
    print(output.shape)
