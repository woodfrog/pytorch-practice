import torch
from torchvision import models


class VGG16(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(VGG16, self).__init__()
        # check https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py for source code
        features = models.vgg16(pretrained=True).features
        extractor_settings = [(0, 4), (4, 9), (9, 16), (16, 23)]
        self.extractors = []

        for setting in extractor_settings:
            extractor = torch.nn.Sequential()
            for x in range(setting[0], setting[1]):
                extractor.add_module(str(x), features[x])
            self.extractors.append(extractor)

        if not requires_grad:
            print(len(list(self.parameters())))
            for para in self.parameters():
                para.require_grad = False

    def forward(self, X):
        extracted_features = []
        h = X
        for extractor in self.extractors:
            extractor.cuda()  # the outside cuda() doesn't work? have to put one extra here?
            h = extractor(h)
            extracted_features.append(h)
        return extracted_features

if __name__ == '__main__':
    vgg = VGG16()
