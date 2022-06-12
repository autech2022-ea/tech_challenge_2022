import numpy as np
import random
from PIL import Image
from torchvision import transforms
# Expanded from https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# Expanded from https://note.nkmk.me/en/python-pillow-add-margin-expand-canvas/


def expand2square(pil_img, shift=None, background_color=None):
    """
    Expand a PIL image to fill a perfect square, without modifying the original
    aspect ratio of the content. At the same time, the original content
    can be slightly shifted in X and/or Y (kind of augmentation).

    :param PIL.Image pil_img: the image to expand
    :param int shift: the number of max pixels that the image can be randomly shifted, or None if no shift
    :param tuple background_color: The PIL color to fill the square
    :return: The expanded, squared PIL.Image
    """
    width, height = pil_img.size
    if not background_color:
        background_color = pil_img.getpixel((0, 0))

    shift_x, shift_y = 0, 0
    if shift:
        shift_x = np.random.randint(-shift, shift)
        shift_y = np.random.randint(-shift, shift)

    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (shift_x, ((width - height) // 2) + shift_y))
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, (((height - width) // 2) + shift_x, shift_y))
    return result


class SmartPadding(object):
    """
    Pad an image to make it squared, while maintaining the original content
    in the center. Additionally, the content may be randomly shifted
    few pixels in the x and/or y axis, as a random augmentation.
    """
    def __init__(self, output_size, shift):
        """
        :param int output_size: Target size. The image will be resized to be output_size x output_size
        :param int shift: the number of max pixels that the image can be randomly shifted, or None if no shift
        """
        assert isinstance(output_size, int)
        self.output_size = output_size
        self.shift = shift

    def __call__(self, image):
        image = expand2square(image, self.shift)
        return image.resize((self.output_size, self.output_size))


class RandomColorShift(object):
    """
    Augmentation: Split the image into R, G, B channels. Create a new image merging
    three of these channels at random.

    NOTE: This type of transformation is very destructive, it only makes sense
          for certain domains.
    """
    def __call__(self, image):
        r, g, b = image.split()
        new_channels = []
        for _ in range(3):
            new_channels.append(random.choices([r, g, b], k=1)[0])
        return Image.merge('RGB', new_channels)


class ChangeRatio(object):
    """
    Deform the image in both X and Y axis.
    """
    def __init__(self, deformation=0.3):
        """
        :param float deformation: the amount of deformation (from 0 to 1).
            For instance, deformation=0.4 means the new width and height will
            be picked uniformly from (1-0.4, 1+0.4) = (0.6, 1.4)
        """
        assert isinstance(deformation, (int, float))
        self.deformation = deformation

    def __call__(self, sample):
        w, h = sample.size
        ratio_w = np.random.uniform(1 - self.deformation, 1 + self.deformation)
        ratio_h = np.random.uniform(1 - self.deformation, 1 + self.deformation)
        new_w = int(w * ratio_w)
        new_h = int(h * ratio_h)
        return sample.resize((new_w, new_h))


def train_transformations(tile_size):
    return [
      transforms.RandomApply([ChangeRatio()], p=0.5),
      transforms.RandomApply([RandomColorShift()], p=0.3),
      transforms.RandomInvert(p=0.10),
      SmartPadding(tile_size, shift=10),
      transforms.ToTensor(),
      transforms.Normalize((0.4915, 0.4823, 0.4468),
                            (0.2470, 0.2435, 0.2616))
    ]


def val_transformation_list(tile_size):
    return [
        SmartPadding(tile_size, shift=None),  # no random shift in x/y
        transforms.ToTensor(),
        transforms.Normalize((0.4915, 0.4823, 0.4468),
                             (0.2470, 0.2435, 0.2616))
    ]


def get_train_compose(tile_size):
    """
    Return the pipeline to transform images for the training set.
    That means, all the augmentations and transforms/normalization.
    """
    transformations = train_transformations(tile_size)
    return transforms.Compose(transformations)


def get_val_compose(tile_size):
    """
    Return the pipeline to transform images for the validation set.
    That means, NO augmentations, only transforms/normalization.
    """
    return transforms.Compose(val_transformation_list(tile_size))


def get_verify_compose(tile_size):
    """
    Return the pipeline to transform images for the manual verify set.
    That means, NO augmentations, NO transform y to tensor.
    It is designed to leave images as PIL.Image, so its easy to verify
    by a human.
    """
    transformations = train_transformations(tile_size)
    return transforms.Compose(transformations[:-2])
