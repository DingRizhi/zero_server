import os
import shutil
import torch.nn as nn
import re
import cv2
import numpy as np
import torch
from PIL import Image
from torch import optim
from torch.nn import Linear, Sequential, ReLU, CrossEntropyLoss, functional
from torch.nn.parallel.data_parallel import DataParallel
from torch.utils.data import DataLoader
from torch.nn import init
from torchvision import transforms, datasets, models


def model_predict(model1, load_path, img_size, target_folder_path, output_folder_path, guojian, class_id, thres):
    if not os.path.exists(output_folder_path):
        os.mkdir(output_folder_path)
    # define the datasets transforms
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # init model in gpu
    # model.load_state_dict(torch.load(load_path))
    model=torch.load(load_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.cuda()
    model.eval()

    # model1 = torch.load(load_path, map_location=lambda storage, loc: storage)
    # model1.eval()
    print('Model Ready ...')
    for img_file in os.listdir(target_folder_path):
        image = cv2.imread(os.path.join(target_folder_path, img_file))
        if image.shape[2] == 1:
            image = image + np.zeros((image.shape[0], image.shape[1], 3), dtype='uint8')
        img = Image.fromarray(image)
        img = transform(img).unsqueeze(dim=0)
        img = img.cuda()
        output = model(img)
        print('output {}'.format(output))
        output = functional.softmax(output, dim=1)
        print('softmax {}'.format(output))

        _, predict = output.topk(1, dim=1, largest=True)
        cls_id = int(predict)
        score = float(output[0][cls_id])
        print('cls_id {},score{}'.format(cls_id, score))

        # print('output {}, softmax {}, predict {}, cls id {}, score {}'.format(output,))
        if guojian:
            if cls_id == class_id and score > thres:
                target_image_name = img_file.replace('.jpg', '_%d_%.2f.jpg' % (cls_id, score))
                shutil.copy(os.path.join(target_folder_path, img_file),
                            os.path.join(output_folder_path, target_image_name))
                print(f'Detect Image {img_file} in class {cls_id} with score {score}')
        else:
            if cls_id != class_id or score < thres:
                target_image_name = img_file.replace('.jpg', '_%d_%.2f.jpg' % (cls_id, score))
                shutil.copy(os.path.join(target_folder_path, img_file),
                            os.path.join(output_folder_path, target_image_name))
                print(f'Detect Image {img_file} in class {cls_id} with score {score}')


def basic_classifier_resnet18(num_classes, pretrained_path=None):
    model = models.resnet18(pretrained=False)
    if pretrained_path is not None:
        model.load_state_dict(torch.load(pretrained_path))
    model.fc = Linear(512, num_classes)
    return model


def basic_classifier_resnet18_v2(num_classes, pretrained_path=None):
    model = Resnet18(num_classes)
    if pretrained_path is not None:
        model.load_state_dict(torch.load(pretrained_path))
    return model


def basic_classifier_resnet50(num_classes, pretrained_path=None):
    model = models.resnet50(pretrained=False)
    if pretrained_path is not None:
        model.load_state_dict(torch.load(pretrained_path))
    model.fc = Sequential(
        Linear(2048, 128),
        ReLU(inplace=True),
        Linear(128, num_classes)
    )
    return model


def basic_classifier_mobilenetv2(num_classes, pretrained_path=None):
    model = models.mobilenet_v2(pretrained=False)
    if pretrained_path is not None:
        model.load_state_dict(torch.load(pretrained_path))
    model.classifier[1] = Linear(in_features=model.classifier[1].in_features, out_features=num_classes)
    return model

def basic_classifier_mobilenetv3(num_classes, pretrained_path=None):
    model = models.mobilenet_v3_large(pretrained=False)
    if pretrained_path is not None:
        model.load_state_dict(torch.load(pretrained_path))
    model.classifier[3] = Linear(in_features=model.classifier[3].in_features, out_features=num_classes)
    return model

def basic_classifier_mobilenetv3_small(num_classes, pretrained_path=None):
    model = models.mobilenet_v3_small(pretrained=False)
    if pretrained_path is not None:
        model.load_state_dict(torch.load(pretrained_path))
    model.classifier[3] = Linear(in_features=model.classifier[3].in_features, out_features=num_classes)
    return model

def basic_classifier_densenet121(num_classes, pretrained_path=None):
    model = models.densenet121(pretrained=False)
    model.classifier.out_features = num_classes
    if pretrained_path is not None:
        pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(pretrained_path)
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
        # model.load_state_dict(torch.load(pretrained_path))
    model.classifier.out_features = num_classes
    return model

def mobilenet_new(num_classes, pretrained_path=None):
    model = models.mobilenet_v2(pretrained=True)
    for parameter in model.parameters():
        parameter.requires_grad = False
    fc = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(1280, num_classes),
    )
    model.classifier = fc
    return model
def basic_classifier_efficientnet_b0(num_classes, pretrained_path=None):
    model = models.efficientnet_b0(pretrained=False)
    if pretrained_path is not None:
        model.load_state_dict(torch.load(pretrained_path))
    model.classifier[1] = Linear(in_features=model.classifier[1].in_features, out_features=num_classes)
    return model

def train(data_dir, img_size, model, device_ids, batch_size, lr_sets, num_epoch, save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # Transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    }
    # Datasets & DataLoaders
    dataset = {
        'train': datasets.ImageFolder(os.path.join(data_dir, 'train'), data_transforms['train']),
        'val': datasets.ImageFolder(os.path.join(data_dir, 'val'), data_transforms['val'])
    }
    dataloaders = {
        'train': DataLoader(dataset['train'], batch_size=batch_size, shuffle=True, num_workers=4),
        'val': DataLoader(dataset['val'], batch_size=batch_size, num_workers=4)
    }
    print('Data Ready ......')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        model = DataParallel(model, device_ids=device_ids)
    model.to(device)
    print(f'Model to Device {device}')
    lr, momentum, internal, decay = lr_sets['lr'], lr_sets['momentum'], lr_sets['internal'], lr_sets['decay']
    criterion = CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    # best_acc = 0
    # best_state = None
    # Train and Val per epoch
    for epoch in range(num_epoch):
        print(f'Epoch {epoch + 1}/{num_epoch}')
        print('-' * 10)
        # Learning Rate Decay
        if (epoch + 1) % internal == 0:
            decayed_lr = lr * (decay ** ((epoch + 1) // internal))
            for param_group in optimizer.param_groups:
                param_group['lr'] = decayed_lr
        train_loss = 0.0
        train_corrects = 0
        val_corrects = 0
        # Train
        model.train()
        for inputs, labels in dataloaders['train']:
            print('input {}, label {}'.format(inputs, labels))
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
            train_corrects += torch.sum(torch.max(outputs, 1)[1] == labels.data)
        epoch_loss = train_loss / len(dataset['train'])
        epoch_acc = train_corrects / len(dataset['train'])
        model.eval()
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            val_corrects += torch.sum(torch.max(outputs, 1)[1] == labels.data)
        val_acc = val_corrects / len(dataset['val'])
        # if val_acc >= best_acc:
        #     best_acc = val_acc
        #     best_state = model.state_dict()
        print(f'Train Loss: {epoch_loss}, Train Acc: {epoch_acc}, Val Acc: {val_acc}')
        torch.save(model.state_dict(), os.path.join(save_path, f'{epoch + 1}.pt'))
    print(f'Model saved in {save_path}')


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)


class Resnet18(nn.Module):

    def __init__(self, class_num):
        super(Resnet18, self).__init__()
        self.base_model = models.resnet18(pretrained=False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        classifier1 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=0.5),
        )
        classifier1.apply(weights_init_kaiming)

        classifier2 = nn.Sequential(
            nn.Linear(256, class_num),
        )
        classifier2.apply(weights_init_classifier)

        self.classifier1 = classifier1
        self.classifier2 = classifier2

    def forward(self, x):
        for name, module in self.base_model._modules.items():
            if name == 'avgpool':
                break
            x = module(x)
        x = self.avgpool(x)
        # features = torch.squeeze(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier1(x)
        x = self.classifier2(x)
        return x

    def outputs(self, args):
        out1, landa = args
        return out1


if __name__ == '__main__':
    train(data_dir=r'/home/adt/code/transfer_learning/data/hymenoptera_data',
          img_size=128,
          model=basic_classifier_mobilenetv2(num_classes=2),
          device_ids=[0],
          batch_size=64,
          lr_sets={'lr': 0.001, 'momentum': 0.9, 'internal': 30, 'decay': 0.1},
          num_epoch=40,
          save_path=r'./weights/mo_neets_0707')
