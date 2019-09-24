import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
from PIL import Image
import json

from imagenet1000_clsidx_to_labels import id2cls_imagenet1000

class ImageSet(Dataset):

    def __init__(self, root, img_size=(224, 224)):

        self.root = root
        self.files = os.listdir(self.root)
        self.transformer = T.Compose([
            T.Resize(img_size),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        filename = self.files[index]
        path = os.path.join(self.root, filename)

        img = Image.open(path)
        img = self.transformer(img)

        return img, filename

def deploy(root_path):

    # 0. path
    data_root = os.path.join(root_path, 'images')
    result_root = os.path.join(root_path, 'results')

    # 1. dataset
    dataset = ImageSet(data_root)

    # 2. model
    net = torchvision.models.resnet18(pretrained=False)
    net.load_state_dict(torch.load('weights/resnet18-5c106cde.pth'))
    if torch.cuda.is_available():
        net = net.cuda()
    net.eval()

    # 3. loop
    result = {}
    for i, (img, filename) in enumerate(dataset):

        print('[{}/{}] {}'.format(i, len(dataset), filename))

        # input
        img = img.unsqueeze(dim=0)
        if torch.cuda.is_available():
            img = img.cuda()

        # forward
        prob = net(img)
        prob = prob.softmax(dim=1)
        topk_prob, topk_cls_id = prob[0].topk(k=5, dim=0)

        # parse result
        result[filename] = '{:.2f}% [{}], {:.2f}% [{}], {:.2f}% [{}], {:.2f}% [{}], {:.2f}% [{}]'.format(
            topk_prob[0]*100, id2cls_imagenet1000[topk_cls_id[0].item()], 
            topk_prob[1]*100, id2cls_imagenet1000[topk_cls_id[1].item()], 
            topk_prob[2]*100, id2cls_imagenet1000[topk_cls_id[2].item()], 
            topk_prob[3]*100, id2cls_imagenet1000[topk_cls_id[3].item()], 
            topk_prob[4]*100, id2cls_imagenet1000[topk_cls_id[4].item()] )

    # 4. serilization
    if not os.path.exists(result_root):
        os.mkdir(result_root)
    with open(os.path.join(result_root, 'results.json'), 'w') as f:
        json.dump(result, f)

if __name__=='__main__':

    if torch.cuda.is_available():
        print('Use NVIDIA CUDA')
    else:
        print('NVIDIA CUDA is NOT FOUND')

    deploy(root_path='data')