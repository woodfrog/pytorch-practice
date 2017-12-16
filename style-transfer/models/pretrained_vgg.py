import torch
from torchvision import models


class VGG16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()
        # check https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py for source code
        features = models.vgg16(pretrained=True).features
        extractor_settings = [(0, 4), (4, 9), (9, 16), (16, 23)]
        self.extractor1 = torch.nn.Sequential()
        self.extractor2 = torch.nn.Sequential()
        self.extractor3 = torch.nn.Sequential()
        self.extractor4 = torch.nn.Sequential()
        self.extractors = [self.extractor1, self.extractor2, self.extractor3, self.extractor4]

        # configure every extractor
        for setting, extractor in zip(extractor_settings, self.extractors):
            for x in range(setting[0], setting[1]):
                extractor.add_module(str(x), features[x])

        if not requires_grad:
            for para in self.parameters():
                para.require_grad = False

    def forward(self, X):
        extracted_features = []
        h = X

        for extractor in self.extractors:
            h = extractor(h)
            extracted_features.append(h)
        return extracted_features

        # h = self.extractor1(X)
        # h_relu1_2 = h
        # h = self.extractor2(h)
        # h_relu2_2 = h
        # h = self.extractor3(h)
        # h_relu3_3 = h
        # h = self.extractor4(h)
        # h_relu4_3 = h
        # out = [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3]
        # return out


if __name__ == '__main__':
    vgg = VGG16()
