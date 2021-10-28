import torch
import torch.nn as nn
from torchvision import models, transforms
import clip


class Resnet18(object):
    '''
    pretrained Resnet18 from torchvision
    '''

    def __init__(self, args, eval=True, share_memory=False, use_conv_feat=True):
        self.model = models.resnet18(pretrained=True)

        if args.gpu:
            self.model = self.model.to(torch.device('cuda'))

        if eval:
            self.model = self.model.eval()

        if share_memory:
            self.model.share_memory()

        if use_conv_feat:
            self.model = nn.Sequential(*list(self.model.children())[:-2])

        self.output_channels = 512

    def extract(self, x):
        return self.model(x)


class Resnet50(object):
    '''
    pretrained Resnet50 from torchvision
    '''

    def __init__(self, args, eval=True, share_memory=False, use_conv_feat=True):
        self.model = models.resnet50(pretrained=True)

        if args.gpu:
            self.model = self.model.to(torch.device('cuda'))

        if eval:
            self.model = self.model.eval()

        if share_memory:
            self.model.share_memory()

        if use_conv_feat:
            self.model = nn.Sequential(*list(self.model.children())[:-2])

        self.output_channels = 2048


class ResnetCLIP(object):
    '''
    pretrained Resnet50 from clip
    '''
    def __init__(self, args, eval=True, share_memory=False, use_conv_feat=True):
        model, preprocess = clip.load(
            "RN50",
            device=('cuda' if args.gpu else 'cpu')
        )
        self.model = model.visual
        self.transform = preprocess

        if eval:
            self.model = self.model.eval()

        if share_memory:
            self.model.share_memory()

        if use_conv_feat:
            self.model.attnpool = nn.Identity()

        self.output_channels = 2048

    def extract(self, x):
        return self.model(x)


class MaskRCNN(object):
    '''
    pretrained MaskRCNN from torchvision
    '''

    def __init__(self, args, eval=True, share_memory=False, min_size=224):
        self.model = models.detection.maskrcnn_resnet50_fpn(pretrained=True, min_size=min_size)
        self.model = self.model.backbone.body
        self.feat_layer = 3

        if args.gpu:
            self.model = self.model.to(torch.device('cuda'))

        if eval:
            self.model = self.model.eval()

        if share_memory:
            self.model.share_memory()

        self.output_channels = 2048

    def extract(self, x):
        features = self.model(x)
        return features[self.feat_layer]


class Resnet(object):

    def __init__(self, args, eval=True, share_memory=False, use_conv_feat=True):
        self.model_type = args.visual_model
        self.gpu = args.gpu

        # normalization transform
        self.transform = self.get_default_transform()

        # choose model type
        if self.model_type == "maskrcnn":
            self.resnet_model = MaskRCNN(args, eval, share_memory)
        elif self.model_type == 'resnet18':
            self.resnet_model = Resnet18(args, eval, share_memory, use_conv_feat=use_conv_feat)
        elif self.model_type == 'resnet50':
            self.resnet_model = Resnet50(args, eval, share_memory, use_conv_feat=use_conv_feat)
        elif self.model_type == 'resnet50_clip':
            self.resnet_model = ResnetCLIP(args, eval, share_memory, use_conv_feat=use_conv_feat)
            self.transform = self.resnet_model.transform

    @staticmethod
    def get_default_transform():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])

    def featurize(self, images, batch=32):
        images_normalized = torch.stack([self.transform(i) for i in images], dim=0)
        if self.gpu:
            images_normalized = images_normalized.to(torch.device('cuda'))

        out = []
        with torch.set_grad_enabled(False):
            for i in range(0, images_normalized.size(0), batch):
                b = images_normalized[i:i+batch]
                out.append(self.resnet_model.extract(b))
        return torch.cat(out, dim=0)