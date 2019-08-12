import numpy as np
import math
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import PIL
from imgaug import augmenters as iaa

# transforms = iaa.SomeOf(3, [
#     iaa.AdditiveGaussianNoise(scale=0.2*255),
#     iaa.GaussianBlur((0.0, 2.0)),
#     iaa.Affine(
#         scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
#         translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
#         rotate=(-5, 5),
#         shear=(-2, 2),
#         cval=(255, 255),
#     ),
#     iaa.SaltAndPepper(.05)
# ])

class ImgAugTransform:
    def __init__(self, img_size=32):
        t = .05
        self.transforms = iaa.Sequential([
            iaa.Resize(img_size),
            iaa.SomeOf(2, [
#                 iaa.Affine(
#                     scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
#                     translate_percent={"x": (-t, t), "y": (-t, t)},
#                     rotate=(-8, 8),
#                     shear=(-4, 4),
#                     cval=(255, 255),
#                 ),
#                 iaa.AdditiveGaussianNoise(scale=0.1*255),
#                 iaa.GaussianBlur((0.0, 2.0)),
                iaa.SaltAndPepper(.1)
            ])
        ])
    def __call__(self, img):
        img = np.array(img)
        return self.transforms.augment_image(img)

def make_loader(batch_size, img_size=32):
    
    dataset = datasets.ImageFolder('./dataset', transform=transforms.Compose([
        ImgAugTransform(img_size),
        lambda x: PIL.Image.fromarray(x),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ]))

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
    ), dataset
