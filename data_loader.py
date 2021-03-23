import numpy as np
from PIL import Image, ImageChops
from torchvision import transforms
import random
import torch
import torchvision.datasets as datasets
import torch.utils.data as data


class SYSUData(data.Dataset):
    def __init__(self, data_dir,  transform=None, colorIndex = None, thermalIndex = None):
        
        # Load training images (path) and labels
        train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')
        
        # RGB format
        self.train_color_image   = train_color_image
        self.train_thermal_image = train_thermal_image
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)
        
        
class RegDBData(data.Dataset):
    def __init__(self, data_dir, trial, transform=None, colorIndex = None, thermalIndex = None):
        # Load training images (path) and labels
        train_color_list   = data_dir + 'idx/train_visible_{}'.format(trial)+ '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial)+ '.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)
        
        train_color_image = []
        for i in range(len(color_img_file)):
            img = Image.open(data_dir+ color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
        train_color_image = np.array(train_color_image) 
        
        train_thermal_image = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir+ thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)
        train_thermal_image = np.array(train_thermal_image)
        
        # RGB format
        self.train_color_image = train_color_image  
        self.train_color_label = train_color_label
        
        # RGB format
        self.train_thermal_image = train_thermal_image
        self.train_thermal_label = train_thermal_label
        
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)

        return img1, img2, target1, target2

    def __len__(self):
        return len(self.train_color_label)
        
class TestData(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size = (224,224)):
        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform

    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        return img1, target1

    def __len__(self):
        return len(self.test_image)
        
def load_data(input_data_path ):
    with open(input_data_path) as f:
        data_file_list = open(input_data_path, 'rt').read().splitlines()
        # Get full list of image and labels
        file_image = [s.split(' ')[0] for s in data_file_list]
        file_label = [int(s.split(' ')[1]) for s in data_file_list]
        
    return file_image, file_label


class SYSUData2(data.Dataset):
    def __init__(self, data_dir,  transform=None, colorIndex = None, thermalIndex = None):
        
        # Load training images (path) and labels
        train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')
        
        # RGB format
        self.train_color_image   = train_color_image
        self.train_thermal_image = train_thermal_image
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        modality_rgb = torch.tensor([1,0]).float()
        modality_ir = torch.tensor([0,1]).float()
        return img1, img2, target1, target2, modality_rgb, modality_ir
    



class TestData2(data.Dataset):
    def __init__(self, test_img_file, test_label, transform=None, img_size = (224,224)):
        test_image = []
        for i in range(len(test_img_file)):
            img = Image.open(test_img_file[i])
            img = img.resize((img_size[0], img_size[1]), Image.ANTIALIAS)
            pix_array = np.array(img)
            test_image.append(pix_array)
        test_image = np.array(test_image)
        self.test_image = test_image
        self.test_label = test_label
        self.transform = transform
    def __getitem__(self, index):
        img1,  target1 = self.test_image[index],  self.test_label[index]
        img1 = self.transform(img1)
        img1 = self.normalizeR(img1)
        return img1, target1
    
class RandomCrop(object):
    def __init__(self, image_w, image_h):
        self.image_h = image_h
        self.image_w = image_w
    def __call__(self, sample):
        image, thermal, label = sample['image'], sample['thermal'], sample['label']

        return {'image': image, 'thermal': thermal, 'label': label}

class RandomFlip(object):
    def __call__(self, sample):
        image, thermal, label = sample['image'], sample['thermal'], sample['label']
        if random.random() > 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            thermal = thermal.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            image   = image.transpose(Image.FLIP_TOP_BOTTOM)
            thermal = thermal.transpose(Image.FLIP_TOP_BOTTOM)
            label   = label.transpose(Image.FLIP_TOP_BOTTOM)
        return {'image': image, 'thermal': thermal, 'label': label}    
    
class SYSUData3(data.Dataset):
    def __init__(self, data_dir,  transform=None, colorIndex = None, thermalIndex = None):
        
        # Load training images (path) and labels
        train_color_image = np.load(data_dir + 'train_rgb_resized_img.npy')
        self.train_color_label = np.load(data_dir + 'train_rgb_resized_label.npy')

        train_thermal_image = np.load(data_dir + 'train_ir_resized_img.npy')
        self.train_thermal_label = np.load(data_dir + 'train_ir_resized_label.npy')
        
        
        self.train_color_mask = np.load(data_dir + 'train_RGB_img_mask.npy')
        self.train_thermal_mask = np.load(data_dir + 'train_IR_img_mask.npy')
        
        # RGB format

        self.train_color_image   = train_color_image
        self.train_thermal_image = train_thermal_image
        self.cIndex = colorIndex
        self.tIndex = thermalIndex
        
        
        self.image_w = 144
        self.image_h = 288
        self.transform = transform
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        self.totensor = transforms.ToTensor()
        self.topil = transforms.ToPILImage()
        self.pad = transforms.Pad(10)
        self.resize = transforms.Resize((self.image_h,self.image_w))
    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        mask1, mask2 = self.train_color_mask[self.cIndex[index]],  self.train_thermal_mask[self.tIndex[index]]
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        
        
        mask1 = self.resize(self.topil(mask1))
        mask2 = self.resize(self.topil(mask2))  
        mask1 = self.pad(mask1)
        mask2 = self.pad(mask2)
        ##随机crop
        ##RGB image
        w , h = img1.size
        i = random.randint(0, w - self.image_w)
        j = random.randint(0, h - self.image_h)
        img1  = img1.crop((i, j, i + self.image_w, j + self.image_h))
        mask1 = mask1.crop((i, j, i + self.image_w, j + self.image_h))
        ##IR image
        w , h = img2.size
        i = random.randint(0, w - self.image_w)
        j = random.randint(0, h - self.image_h)
        img2  = img2.crop((i, j, i + self.image_w, j + self.image_h))
        mask2 = mask2.crop((i, j, i + self.image_w, j + self.image_h))        

        ##随机翻转
        if random.random() > 0.5:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            mask1 = mask1.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
            mask2 = mask2.transpose(Image.FLIP_LEFT_RIGHT)
        
        img1 = self.totensor(img1)
        img2 = self.totensor(img2)
        mask1 = self.totensor(mask1)
        mask2 = self.totensor(mask2)
        
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)
        return img1, img2, target1, target2, mask1, mask2

    def __len__(self):
        return len(self.train_color_label)

class RegDBData2(data.Dataset):
    def __init__(self, data_dir, trial, transform=None, colorIndex = None, thermalIndex = None):
        # Load training images (path) and labels
        train_color_list   = data_dir + 'idx/train_visible_{}'.format(trial)+ '.txt'
        train_thermal_list = data_dir + 'idx/train_thermal_{}'.format(trial)+ '.txt'

        color_img_file, train_color_label = load_data(train_color_list)
        thermal_img_file, train_thermal_label = load_data(train_thermal_list)
        
        train_color_image = []
        train_color_image_mask = []
        for i in range(len(color_img_file)):
            img = Image.open(data_dir+ color_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_color_image.append(pix_array)
            
            img_mask = Image.open(data_dir[0:22]+'mask/'+data_dir[22:] + color_img_file[i])
            img_mask = img_mask.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img_mask)
            train_color_image_mask.append(pix_array)
            
        train_color_image = np.array(train_color_image) 
        train_color_image_mask = np.array(train_color_image_mask)
        
        
        train_thermal_image = []
        train_thermal_image_mask = []
        for i in range(len(thermal_img_file)):
            img = Image.open(data_dir+ thermal_img_file[i])
            img = img.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img)
            train_thermal_image.append(pix_array)

            img_mask = Image.open(data_dir[0:22]+'mask/'+data_dir[22:] + thermal_img_file[i])
            img_mask = img_mask.resize((144, 288), Image.ANTIALIAS)
            pix_array = np.array(img_mask)
            train_thermal_image_mask.append(pix_array)
            
        train_thermal_image = np.array(train_thermal_image)
        train_thermal_image_mask = np.array(train_thermal_image_mask)
        # RGB format
        self.train_color_image = train_color_image  
        self.train_color_image_mask = train_color_image_mask 
        self.train_color_label = train_color_label
        
        # RGB format
        self.train_thermal_image = train_thermal_image
        self.train_thermal_image_mask = train_thermal_image_mask
        self.train_thermal_label = train_thermal_label
        
        self.transform = transform
        self.cIndex = colorIndex
        self.tIndex = thermalIndex

        self.image_w = 144
        self.image_h = 288
        self.transform = transform
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        self.totensor = transforms.ToTensor()
        self.topil = transforms.ToPILImage()
        self.pad = transforms.Pad(10)
        self.resize = transforms.Resize((self.image_h,self.image_w))
    def __getitem__(self, index):

        img1,  target1 = self.train_color_image[self.cIndex[index]],  self.train_color_label[self.cIndex[index]]
        img2,  target2 = self.train_thermal_image[self.tIndex[index]], self.train_thermal_label[self.tIndex[index]]
        mask1, mask2 = self.train_color_image_mask[self.cIndex[index]],  self.train_thermal_image_mask[self.tIndex[index]]
        img1 = self.transform(img1)
        img2 = self.transform(img2)
        
        
        mask1 = self.resize(self.topil(mask1))
        mask2 = self.resize(self.topil(mask2))  
        mask1 = self.pad(mask1)
        mask2 = self.pad(mask2)
        ##随机crop
        ##RGB image
        w , h = img1.size
        i = random.randint(0, w - self.image_w)
        j = random.randint(0, h - self.image_h)
        img1  = img1.crop((i, j, i + self.image_w, j + self.image_h))
        mask1 = mask1.crop((i, j, i + self.image_w, j + self.image_h))
        ##IR image
        w , h = img2.size
        i = random.randint(0, w - self.image_w)
        j = random.randint(0, h - self.image_h)
        img2  = img2.crop((i, j, i + self.image_w, j + self.image_h))
        mask2 = mask2.crop((i, j, i + self.image_w, j + self.image_h))        

        ##随机翻转
        if random.random() > 0.5:
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)
            mask1 = mask1.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() > 0.5:
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
            mask2 = mask2.transpose(Image.FLIP_LEFT_RIGHT)
        
        img1 = self.totensor(img1)
        img2 = self.totensor(img2)
        mask1 = self.totensor(mask1)
        mask2 = self.totensor(mask2)
        
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)
        return img1, img2, target1, target2, mask1, mask2

    def __len__(self):
        return len(self.train_color_label)
    
