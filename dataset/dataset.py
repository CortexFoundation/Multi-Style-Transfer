import os
import glob
import mxnet as mx
import numpy as np
from PIL import Image
import mxnet.gluon.data as data
from utils import tensor_load_rgbimage



class ImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::
    Args:
        root (string): Root directory path.
        img_size: The size of target imgs.
        img_style: Format of imgs.
    """

    def __init__(self, 
                root, 
                img_size,
                ctx, 
                img_style='.jpg'):
        super(ImageFolder, self).__init__()
        self.img_size = img_size
        self.ctx = ctx
        self.root = root
        self.img_style = img_style
        self.imgs = self._get_imgs()
        self._get_transfroms()

    def __getitem__(self, index):
        img = self._load_img(self.imgs[index])
        img = self._transforms(img)
        return img

    def __len__(self):
        return len(self.imgs)

    def __size__(self):
        return self.img_size

    def _get_imgs(self):
        assert os.path.isdir(self.root), "InputError: {} is not a dir".format(self.root)
        imgs_list = glob.glob(self.root + '*' + self.img_style)
        assert len(imgs_list) > 0, "InputError, there is no {} images in {}".format(self.img_style, self.root)
        imgs_list.sort()
        return imgs_list

    def _get_transfroms(self):
        self.scale = Scale(self.img_size)
        self.crop = CenterCrop(self.img_size)

    def _transforms(self, img):
        img = self.scale(img)
        img = self.crop(img)
        img = mx.nd.array(np.array(img).transpose(2, 0, 1).astype('float32'), ctx=self.ctx)
        return img

    def _load_img(self, img_path):
        with open(img_path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')

    def get(self, idx):
        idx = idx%len(self.imgs)
        return mx.nd.expand_dims(self.__getitem__(idx), 0)
    





class Scale(object):
    """Rescale the input PIL.Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size, int) 
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)

class CenterCrop(object):
    """Crops the given PIL.Image at the center.
    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be cropped.
        Returns:
            PIL.Image: Cropped image.
        """
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return img.crop((x1, y1, x1 + tw, y1 + th))


class StyleLoader():
    def __init__(self, style_folder, style_size, ctx):
        self.folder = style_folder
        self.style_size = style_size
        self.files = os.listdir(style_folder)
        assert(len(self.files) > 0)
        self.ctx = ctx

    def get(self, i):
        idx = i%len(self.files)
        filepath = os.path.join(self.folder, self.files[idx])
        style = tensor_load_rgbimage(filepath, self.ctx, self.style_size)
        return style

    def size(self):
        return len(self.files)