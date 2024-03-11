from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import json
from easydict import EasyDict
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.model_selection import train_test_split

class Biopsy_Dataset(Dataset):

    def __init__(self, config,transform=None):
        #super(Biopsy_Dataset).__init__()
        self.transform = transform
        self.config = config
        self.tile_folders = sorted([file for file in Path(config.tile_path).glob('*')])
        self.image_folders_HE = []
        self.image_folders_MUC = []
        for tile_folder in self.tile_folders:
            image_folders_HE_current = sorted([file for file in tile_folder.glob('*HE*')])
            self.image_folders_HE.extend(image_folders_HE_current)
            image_folders_MUC_current = sorted([file for file in tile_folder.glob('*MUC*')])
            self.image_folders_MUC.extend(image_folders_MUC_current)
        self.image_files_HE = sorted([file for directory in self.image_folders_HE for file in directory.glob('*')])
        self.image_files_MUC = sorted([file for directory in self.image_folders_MUC for file in directory.glob('*')])
        #not needed for cyclegan
        #assert len(self.image_files_HE) == len(self.image_files_MUC), "You need equally much HE and MUC images"

    def __getitem__(self, idx):
        image_path_HE = self.image_files_HE[idx]
        image_path_MUC = self.image_files_MUC[idx]
        image_HE = Image.open(image_path_HE).convert('RGB')
        image_MUC = Image.open(image_path_MUC).convert('RGB')
        #labels = [float(value) for value in self.data.iloc[idx, 1:].values]

        # Apply transformations if specified
        if self.transform:
            image_HE = self.transform(image_HE)
            image_MUC = self.transform(image_MUC)
            if check_permute(image_HE):
                image_HE = image_HE.permute(1, 2, 0)
            if check_permute(image_MUC):
                image_MUC = image_MUC.permute(1, 2, 0)

        #return {'image_HE':image_HE, 'image_MUC' : image_MUC, 'image_path_HE': image_path_HE, 'image_path_MUC': image_path_MUC}
        return {'image_HE':image_HE, 'image_MUC' : image_MUC}

    def __len__(self):
        return len(self.image_files_MUC)


def check_permute(image):
    #If image is in (C,H,W) format, it needs to be rearranged to (H,W,C) format
    if image.shape[0] < image.shape[1] and image.shape[0] < image.shape[2]:
        return True
    else:
        return False

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

class Biopsy_Dataloader:
    #def __init__(self,dataset,batch_size = 32,test_size = 0.1, val_size = 0.1):
        #self.dataset = dataset
    def __init__(self,config,dataset,batch_size = 32,test_size = 0.1, val_size = 0.1):
        self.config = config
        self.dataset = dataset
        self.batch_size = batch_size

        patients = [str(patient_folder.stem) for patient_folder in dataset.image_folders_HE]

        train_patients, test_val_patients = train_test_split(patients, test_size=test_size+val_size, random_state=42)
        val_patients, test_patients = train_test_split(test_val_patients, test_size=test_size/(test_size+val_size), random_state=42)

        train_indices = [idx for idx, patient_folder in enumerate(dataset.image_folders_HE) if
                         patient_folder.stem in train_patients]
        val_indices = [idx for idx, patient_folder in enumerate(dataset.image_folders_HE) if
                       patient_folder.stem in val_patients]
        test_indices = [idx for idx, patient_folder in enumerate(dataset.image_folders_HE) if
                        patient_folder.stem in test_patients]

        # Define samplers and dataloaders
        self.train_sampler = SubsetRandomSampler(train_indices)
        self.val_sampler = SubsetRandomSampler(val_indices)
        self.test_sampler = SubsetRandomSampler(test_indices)

        self.train_loader = DataLoader(dataset, batch_size=batch_size, sampler=self.train_sampler)
        self.val_loader = DataLoader(dataset, batch_size=batch_size, sampler=self.val_sampler)
        self.test_loader = DataLoader(dataset, batch_size=batch_size, sampler=self.test_sampler)

    def get_train_loader(self):
        return self.train_loader

    def get_val_loader(self):
        return self.val_loader

    def get_test_loader(self):
        return self.test_loader

#Give an example plot of what a set of HE - MUC image looks like
if __name__ == "__main__":
    config_filename = r"/esat/biomeddata/kkontras/r0786880/models/remote/configuration.json"
    with open(config_filename, 'r') as config_json:
        a = json.load(config_json)
        config = EasyDict(a)
    biopsy_dataset = Biopsy_Dataset(config, transform)
    plt.imshow(biopsy_dataset[0]["image_HE"])
    plt.show()
    plt.imshow(biopsy_dataset[0]["image_MUC"])
    plt.show()