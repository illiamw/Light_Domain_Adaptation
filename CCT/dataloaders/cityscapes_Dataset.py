import random, math
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from scipy import ndimage
from math import ceil
import os
from torch.utils.data import DataLoader


IMG_MEAN = np.array((104.00698793, 116.66876762,
                     122.67891434), dtype=np.float32)
NUM_CLASSES = 3

# For visualization
label_colours = list(map(tuple, [
    [128, 64, 128],
    [255, 0, 0],
    [255, 255, 0],
    [0, 0, 0],  # the color of ignored label
]))

# Labels
ignore_label = -1
cityscapes_id_to_trainid = {
    -1: ignore_label,
    0: ignore_label,
    1: ignore_label,
    2: ignore_label,
    3: ignore_label,
    4: 2,
    5: 2,
    6: 1,
    7: 0,
    8: 1,
    9: 1,
    10: 1,
    11: 2,
    12: 2,
    13: 2,
    14: 2,
    15: 2,
    16: 2,
    17: 2,
    18: 2,
    19: 2,
    20: 2,
    21: 2,
    22: 1,
    23: 1,
    24: 2,
    25: 2,
    26: 2,
    27: 2,
    28: 2,
    29: 2,
    30: 2,
    31: 2,
    32: 2,
    33: 2
}


# Names
name_classes = [
    'Navegavel',
    'Inavegavel',
    'Obstaculo',
    'Nulo'
]

class CityScape(Dataset):
    def __init__(self, data_dir, split, mean, std, ignore_index, base_size=None, augment=True, val=False,
                jitter=False, use_weak_lables=False, weak_labels_output=None, crop_size=None, scale=False, flip=False, rotate=False,
                blur=False, return_id=False, n_labeled_examples=None):
        self.num_classes = NUM_CLASSES
        self.root = data_dir
        self.split = split
        self.mean = mean
        self.std = std
        self.augment = augment
        self.crop_size = crop_size
        self.jitter = jitter
        self.image_padding = (np.array(mean)*255.).tolist()
        self.ignore_index = ignore_index
        self.return_id = return_id
        self.n_labeled_examples = n_labeled_examples
        self.val = val

        self.use_weak_lables = use_weak_lables
        self.weak_labels_output = weak_labels_output

        if self.augment:
            self.base_size = base_size
            self.scale = scale
            self.flip = flip
            self.rotate = rotate
            self.blur = blur

        self.jitter_tf = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean, std)

        self.id_to_trainid = cityscapes_id_to_trainid

       
        if self.split  == 'train_unsupervised':
            self._set_files(self.root,'train')
        elif self.split == 'train_supervised':
            self._set_files(self.root,'train')
        elif self.split == 'val':
            self._set_files(self.root,'val')


        cv2.setNumThreads(0)

    def _set_files(self,
        root,
        split= "train",
        mode= "fine",
        target_type = "semantic",
        transform  = None,
        target_transform = None,
        transforms = None):

        self.root = root
        self.mode = "gtFine" if mode == "fine" else "gtCoarse"
        self.images_dir = os.path.join(self.root, "leftImg8bit", split)
        self.targets_dir = os.path.join(self.root, self.mode, split)
        self.target_type = target_type
        self.split = split
        self.images = []
        self.targets = []
        

        if not isinstance(target_type, list):
            self.target_type = [target_type]
        

        for city in os.listdir(self.images_dir):
            img_dir = os.path.join(self.images_dir, city)
            target_dir = os.path.join(self.targets_dir, city)
            for file_name in os.listdir(img_dir):
                target_types = []
                for t in self.target_type:
                    target_name = "{}_{}".format(
                        file_name.split("_leftImg8bit")[0], self._get_target_suffix(self.mode, t)
                    )
                    target_types.append(os.path.join(target_dir, target_name))

                self.images.append(os.path.join(img_dir, file_name))
                self.targets.append(target_types)

        #balanciador de carga ----------- HardCoder --------------------
        if self.split == 'train':

            aux = []
            aux2 = []
            tam = len(self.images)
            j = 0
            for i in range(self.n_labeled_examples):
                
                aux.append(self.images[j])
                aux2.append(self.targets[j])
                j+=1
                if j == tam:
                    j=0
            self.images = aux
            self.targets = aux2
        print(f"{self.split}: {len(self.images), {len(self.targets)}}")

    def _get_target_suffix(self, mode: str, target_type: str) -> str:
        if target_type == "instance":
            return f"{mode}_instanceIds.png"
        elif target_type == "semantic":
            return f"{mode}_labelIds.png"
        elif target_type == "color":
            return f"{mode}_color.png"
        else:
            return f"{mode}_polygons.json"


    def _rotate(self, image, label):
        # Rotate the image with an angle between -10 and 10
        h, w, _ = image.shape
        angle = random.randint(-10, 10)
        center = (w / 2, h / 2)
        rot_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        image = cv2.warpAffine(image, rot_matrix, (w, h), flags=cv2.INTER_CUBIC)#, borderMode=cv2.BORDER_REFLECT)
        label = cv2.warpAffine(label, rot_matrix, (w, h), flags=cv2.INTER_NEAREST)#,  borderMode=cv2.BORDER_REFLECT)
        return image, label            

    def _crop(self, image, label):   
        # Padding to return the correct crop size
        if (isinstance(self.crop_size, list) or isinstance(self.crop_size, tuple)) and len(self.crop_size) == 2:
            crop_h, crop_w = self.crop_size 
        elif isinstance(self.crop_size, int):
            crop_h, crop_w = self.crop_size, self.crop_size 
        else:
            raise ValueError

        h, w, _ = image.shape
        pad_h = max(crop_h - h, 0)
        pad_w = max(crop_w - w, 0)
        pad_kwargs = {
            "top": 0,
            "bottom": pad_h,
            "left": 0,
            "right": pad_w,
            "borderType": cv2.BORDER_CONSTANT,}
        if pad_h > 0 or pad_w > 0:
            image = cv2.copyMakeBorder(image, value=self.image_padding, **pad_kwargs)
            label = cv2.copyMakeBorder(label, value=self.ignore_index, **pad_kwargs)

        # Cropping 
        h, w, _ = image.shape
        start_h = random.randint(0, h - crop_h)
        start_w = random.randint(0, w - crop_w)
        end_h = start_h + crop_h
        end_w = start_w + crop_w
        image = image[start_h:end_h, start_w:end_w]
        label = label[start_h:end_h, start_w:end_w]
        return image, label

    def _blur(self, image, label):
        # Gaussian Blud (sigma between 0 and 1.5)
        sigma = random.random() * 1.5
        ksize = int(3.3 * sigma)
        ksize = ksize + 1 if ksize % 2 == 0 else ksize
        image = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT_101)
        return image, label

    def _flip(self, image, label):
        # Random H flip
        if random.random() > 0.5:
            image = np.fliplr(image).copy()
            label = np.fliplr(label).copy()
        return image, label

    def _resize(self, image, label, bigger_side_to_base_size=True):
        if isinstance(self.base_size, int):
            h, w, _ = image.shape
            if self.scale:
                longside = random.randint(int(self.base_size*0.5), int(self.base_size*2.0))
                #longside = random.randint(int(self.base_size*0.5), int(self.base_size*1))
            else:
                longside = self.base_size

            if bigger_side_to_base_size:
                h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h > w else (int(1.0 * longside * h / w + 0.5), longside)
            else:
                h, w = (longside, int(1.0 * longside * w / h + 0.5)) if h < w else (int(1.0 * longside * h / w + 0.5), longside)
            image = np.asarray(Image.fromarray(np.uint8(image)).resize((w, h), Image.BICUBIC))
            label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
            return image, label

        elif (isinstance(self.base_size, list) or isinstance(self.base_size, tuple)) and len(self.base_size) == 2:
            h, w, _ = image.shape
            if self.scale:
                scale = random.random() * 1.5 + 0.5 # Scaling between [0.5, 2]
                h, w = int(self.base_size[0] * scale), int(self.base_size[1] * scale)
            else:
                h, w = self.base_size
            image = np.asarray(Image.fromarray(np.uint8(image)).resize((w, h), Image.BICUBIC))
            label = cv2.resize(label, (w, h), interpolation=cv2.INTER_NEAREST)
            return image, label

        else:
            raise ValueError

    def _val_augmentation(self, image, label):
        if self.base_size is not None:
            image, label = self._resize(image, label)
            image = self.normalize(self.to_tensor(Image.fromarray(np.uint8(image))))
            return image, label

        image = self.normalize(self.to_tensor(Image.fromarray(np.uint8(image))))
        return image, label

    def _augmentation(self, image, label):
        h, w, _ = image.shape

        if self.base_size is not None:
            image, label = self._resize(image, label)

        if self.crop_size is not None:
            image, label = self._crop(image, label)

        if self.flip:
            image, label = self._flip(image, label)

        image = Image.fromarray(np.uint8(image))
        image = self.jitter_tf(image) if self.jitter else image    
        
        return self.normalize(self.to_tensor(image)), label

    def __len__(self):
        return len(self.images)
    
    def id2trainId(self, label, reverse=False, ignore_label=-1):
        label_copy = ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        return label_copy
    
    def _mask_transform(self, gt_image):
        target = np.asarray(gt_image, np.float32)
        target = self.id2trainId(target).copy()
        target = torch.from_numpy(target)

        return target
  
    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")

        image = np.asarray(image, dtype=np.float32)

        targets = []
        for i, t in enumerate(self.target_type):
            if t == "polygon":
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])  # type: ignore[assignment]

            targets.append(target)

        target = tuple(targets) if len(targets) > 1 else targets[0]
        label = target
        label = np.asarray(label, dtype=np.int32)
        if self.val:
            image, label = self._val_augmentation(image, label)
        elif self.augment:
            image, label = self._augmentation(image, label)

        label = self._mask_transform(label)
        return image, label

    def __repr__(self):
        fmt_str = "Dataset: " + self.__class__.__name__ + "\n"
        fmt_str += "    # data: {}\n".format(self.__len__())
        fmt_str += "    Split: {}\n".format(self.split)
        fmt_str += "    Root: {}".format(self.root)
        return fmt_str
    
def decode_labels(mask, num_images=1, num_classes=NUM_CLASSES):
    """Decode batch of segmentation masks.

    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict.

    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    if len(mask.shape) == 2 :
      empty_tensor = torch.empty((1, mask.shape[0], mask.shape[1]), dtype=torch.long)
      empty_tensor[0] = mask
      mask =  empty_tensor

    if isinstance(mask, torch.Tensor):
        mask = mask.data.cpu().numpy()
    n, h, w = mask.shape
    if n < num_images:
        num_images = n
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):

        temp=mask[i].copy()
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, num_classes):
            r[temp == l] = label_colours[l][0]
            g[temp == l] = label_colours[l][1]
            b[temp == l] = label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        rgb = np.stack([r, g, b], axis=-1)
        outputs[i] = rgb
    return torch.from_numpy(outputs)

def inv_preprocess(imgs, num_images=1, img_mean=IMG_MEAN, numpy_transform=False):
    """Inverse preprocessing of the batch of images.

    Args:
      imgs: batch of input images.
      num_images: number of images to apply the inverse transformations on.
      img_mean: vector of mean colour values.
      numpy_transform: whether change RGB to BGR during img_transform.

    Returns:
      The batch of the size num_images with the same spatial dimensions as the input.
    """
    if numpy_transform:
        imgs = flip(imgs, 1)

    def norm_ip(img, min, max):
        img.clamp_(min=min, max=max)
        img.add_(-min).div_(max - min + 1e-5)
    norm_ip(imgs, float(imgs.min()), float(imgs.max()))
    return imgs

def inspect_decode_labels(pred, num_images=1, num_classes=NUM_CLASSES,
                        inspect_split=[0.9, 0.8, 0.7, 0.5, 0.0], inspect_ratio=[1.0, 0.8, 0.6, 0.3]):
    """Decode batch of segmentation masks accroding to the prediction probability.

    Args:
    pred: result of inference.
    num_images: number of images to decode from the batch.
    num_classes: number of classes to predict (including background).
    inspect_split: probability between different split has different brightness.

    Returns:
    A batch with num_images RGB images of the same size as the input. 
    """
    if isinstance(pred, torch.Tensor):
        pred = pred.data.cpu().numpy()
    n, c, h, w = pred.shape
    pred = pred.transpose([0, 2, 3, 1])
    if n < num_images:
        num_images = n
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new('RGB', (w, h))
        pixels = img.load()
        for j_, j in enumerate(pred[i, :, :, :]):
            for k_, k in enumerate(j):
                assert k.shape[0] == num_classes
                k_value = np.max(softmax(k))
                k_class = np.argmax(k)
                for it, iv in enumerate(inspect_split):
                    if k_value > iv:
                        break
                if iv > 0:
                    pixels[k_, j_] = tuple(
                        map(lambda x: int(inspect_ratio[it] * x), label_colours[k_class]))
        outputs[i] = np.array(img)
    return torch.from_numpy(outputs.transpose([0, 3, 1, 2]).astype('float32')).div_(255.0)


class CityscapeDataLoader(DataLoader):
    def __init__(self, kwargs):
        
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]
        self.batch_size = kwargs.pop('batch_size')
        kwargs['mean'] = self.MEAN
        kwargs['std'] = self.STD
        kwargs['ignore_index'] = -1
        try:
            shuffle = kwargs.pop('shuffle')
        except:
            shuffle = False
        num_workers = kwargs.pop('num_workers')

        self.dataset = CityScape(**kwargs)

        self.shuffle = shuffle
        self.nbr_examples = len(self.dataset)
        val_split = None
        if val_split:
            self.train_sampler, self.val_sampler = self._split_sampler(val_split)
        else:
            self.train_sampler, self.val_sampler = None, None

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'num_workers': num_workers,
            'pin_memory': False
        }
        

        super(CityscapeDataLoader, self).__init__(sampler=self.train_sampler, **self.init_kwargs)
