from torch.utils.data import Dataset
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import os

from config import DATA_PATH


class RestrictedImageNet(Dataset):
    classes = {
        'n01532829': 'BIRD',
        'n01558993': 'BIRD',
        'n01843383': 'BIRD',
        'n01855672': 'BIRD',

        'n02089867': 'DOG',
        'n02091244': 'DOG',
        'n02099601': 'DOG',
        'n02101006': 'DOG',
        'n02105505': 'DOG',
        'n02108551': 'DOG',
        'n02108915': 'DOG',
        'n02110063': 'DOG',
        'n02111277': 'DOG',
        'n02114548': 'DOG',
        'n02091831': 'DOG',
        'n02108089': 'DOG',
        'n02110341': 'DOG',
        'n02113712': 'DOG',

        'n02165456': 'INSECT',
        'n02174001': 'INSECT',
        'n02219486': 'INSECT',
        'n01770081': 'INSECT',

        'n02795169': 'CONTAINER',
        'n03127925': 'CONTAINER',
        'n03337140': 'CONTAINER',
        'n02747177': 'CONTAINER',

        'n03272010': 'INSTRUMENT',
        'n03838899': 'INSTRUMENT',
        'n03854065': 'INSTRUMENT',
        'n04515003': 'INSTRUMENT',

        'n03417042': 'VEHICLE',
        'n04146614': 'VEHICLE',
        'n04389033': 'VEHICLE',

        'n04596742': 'FOOD',
        'n07747607': 'FOOD',
        'n03400231': 'FOOD',
        'n07584110': 'FOOD',
        'n07613480': 'FOOD',

        'n01910747': 'SEA_CREATURE',
        'n01981276': 'SEA_CREATURE',
        'n02074367': 'SEA_CREATURE',
        'n02606052': 'SEA_CREATURE',

    }

    def __init__(self, transform_list=None):
        """Dataset class representing representing a restricted ImageNet


        """
        self.df = pd.DataFrame(self.index_data())

        # Index of dataframe has direct correspondence to item in dataset
        self.df = self.df.assign(id=self.df.index.values)

        # Convert arbitrary class names of dataset to ordered 0-(num_speakers - 1) integers
        self.unique_characters = sorted(self.df['class_name'].unique())
        self.class_name_to_id = {self.unique_characters[i]: i for i in range(self.num_classes())}
        self.df = self.df.assign(class_id=self.df['class_name'].apply(lambda c: self.class_name_to_id[c]))

        # Create dicts
        self.datasetid_to_filepath = self.df.to_dict()['filepath']
        self.datasetid_to_class_id = self.df.to_dict()['class_id']

        # Setup transforms
        if transform_list is None:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.75, 1)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),

            ])
        else:
            self.transform = transform_list

    def __getitem__(self, item):
        instance = Image.open(self.datasetid_to_filepath[item])
        instance = self.transform(instance)
        label = self.datasetid_to_class_id[item]
        return instance, label

    def __len__(self):
        return len(self.df)

    def num_classes(self):
        return len(self.df['class_name'].unique())

    @staticmethod
    def index_data():
        """Index a subset by looping through all of its files and recording relevant information.

        # Arguments
            subset: Name of the subset

        # Returns
            A list of dicts containing information about all the image files in a particular subset of the
            miniImageNet dataset
        """
        images = []
        # Quick first pass to find total for tqdm bar

        for root, folders, files in os.walk(DATA_PATH + '/miniImageNet/images_background'):
            if len(files) == 0:
                continue

            class_name = root.split('/')[-1]

            if not class_name in RestrictedImageNet.classes.keys():
                continue

            for f in files:
                images.append({
                    'class_name': RestrictedImageNet.classes[class_name],
                    'filepath': os.path.join(root, f)
                })

        for root, folders, files in os.walk(DATA_PATH + '/miniImageNet/images_evaluation/'):
            if len(files) == 0:
                continue

            class_name = root.split('/')[-1]

            if not class_name in RestrictedImageNet.classes.keys():
                continue

            for f in files:
                images.append({
                    'class_name': RestrictedImageNet.classes[class_name],
                    'filepath': os.path.join(root, f)
                })

        return images
