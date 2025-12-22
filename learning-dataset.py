import os

import torch
import xmltodict
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class VOCDataset(Dataset):
    def __init__(self, image_folder, label_folder, transform, label_transform):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.label_transform = label_transform
        self.img_names = os.listdir(self.image_folder)
        self.classes_list = ["no helmet", "motor", "number", "with helmet"]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert("RGB")
        label_name = img_name.split(".")[0] + ".xml"
        label_path = os.path.join(self.label_folder, label_name)
        with open(label_path, "r", encoding="utf-8") as f:
            label_content = f.read()
        label_dict = xmltodict.parse(label_content)
        objects = label_dict["annotation"]["object"]
        target = []
        for object in objects:
            object_name = object["name"]
            object_class_id = self.classes_list.index(object_name)
            object_xmax = float(object["bndbox"]["xmax"])
            object_ymax = float(object["bndbox"]["ymax"])
            object_xmin = float(object["bndbox"]["xmin"])
            object_ymin = float(object["bndbox"]["ymin"])
            target.extend([object_class_id, object_xmax, object_ymax, object_xmin, object_ymin])
        target = torch.tensor(target)
        if self.transform is not None:
            image = self.transform(image)
        return image, target

class YOLODataset(Dataset):
    def __init__(self, image_folder, label_folder, transform, label_transform):
        self.image_folder = image_folder
        self.label_folder = label_folder
        self.transform = transform
        self.label_transform = label_transform
        self.img_names = os.listdir(self.image_folder)
        self.classes_list = ["no helmet", "motor", "number", "with helmet"]

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_name = self.img_names[index]
        img_path = os.path.join(self.image_folder, img_name)
        image = Image.open(img_path).convert("RGB")
        label_name = img_name.split(".")[0] + ".txt"
        label_path = os.path.join(self.label_folder, label_name)
        with open(label_path, "r", encoding="utf-8") as f:
            label_content = f.read()
        object_infos = label_content.strip().split("\n")
        target = []
        for object_info in object_infos:
            info_list = object_info.strip().split(" ")
            class_id = float(info_list[0])
            center_x = float(info_list[1])
            center_y = float(info_list[2])
            width = float(info_list[3])
            height = float(info_list[4])
            target.extend([class_id, center_x, center_y, width, height])
        target = torch.tensor(target)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


if __name__ == '__main__':
    # train_dataset = VOCDataset("F:\Python\Project\psychic-online-ai\\test-data\\voc\images",
    #                            "F:\Python\Project\psychic-online-ai\\test-data\\voc\Annotations",
    #                            transforms.Compose([transforms.ToTensor()]), None)
    # print(len(train_dataset))
    # print(train_dataset[2])

    train_dataset = YOLODataset("F:\Python\Project\psychic-online-ai\\test-data\\yolo\images",
                               "F:\Python\Project\psychic-online-ai\\test-data\\yolo\labels",
                               transforms.Compose([transforms.ToTensor()]), None)
    print(len(train_dataset))
    print(train_dataset[2])
