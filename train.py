import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import collections

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from retinanet import coco_eval
from retinanet import csv_eval

import csv

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def train():
    loss_hist = collections.deque(maxlen=500)

    if not os.path.exists('logs/{}'.format(parser.depth)):
        os.makedirs('logs/{}'.format(parser.depth))
    file_train = open('logs/{}/train.csv'.format(parser.depth), 'w')
    file_val = open('logs/{}/val.csv'.format(parser.depth), 'w')
    train_writer = csv.writer(file_train)
    val_writer = csv.writer(file_val)

    print('Num training images: {}'.format(len(dataset_train)))

    for epoch_num in range(parser.epochs):
        retinanet.train()

        if use_gpu and torch.cuda.is_available():
            retinanet.module.freeze_bn()
        else:
            retinanet.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()

                if use_gpu and torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].float().cuda(), data['annot'].cuda()])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])
                    
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                print(
                    'Epoch: {} | Iteration: {} | Loss: {:1.5f} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running loss: {:1.5f}'.format(
                        epoch_num, iter_num, float(loss), float(classification_loss), float(regression_loss), np.mean(loss_hist)))

                train_writer.writerow([epoch_num, iter_num, float(loss), float(classification_loss), float(regression_loss), np.mean(loss_hist)])
                file_train.flush()

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue
        
        if not os.path.exists('models/{}'.format(parser.depth)):
            os.makedirs('models/{}'.format(parser.depth))

        if use_gpu and torch.cuda.is_available():
            torch.save(retinanet.module, 'models/{}/{}_retinanet{}_{}.pt'.format(parser.depth, parser.dataset, parser.depth, epoch_num))
        else:
            torch.save(retinanet, 'models/{}/{}_retinanet{}_{}.pt'.format(parser.depth, parser.dataset, parser.depth, epoch_num))

        scheduler.step(np.mean(epoch_loss))

        if parser.dataset == 'coco':
            print('Evaluating COCO dataset')
            stats = coco_eval.evaluate_coco(dataset_val, retinanet, use_gpu=use_gpu)
            val_writer.writerow(stats)
            file_val.flush()

    retinanet.eval()

    file_train.close()
    file_val.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)
    parser.add_argument('--batch_size', help='Size of batch', type=int, default=2)

    parser = parser.parse_args()

    # Create the data loaders
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))
    else:
        raise ValueError('Dataset type not understood (must be coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=parser.batch_size, drop_last=False)
    dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    use_gpu = True

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=True, use_gpu=use_gpu)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_train.num_classes(), pretrained=True, use_gpu=use_gpu)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_train.num_classes(), pretrained=True, use_gpu=use_gpu)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_train.num_classes(), pretrained=True, use_gpu=use_gpu)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_train.num_classes(), pretrained=True, use_gpu=use_gpu)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    if use_gpu and torch.cuda.is_available():
        retinanet = retinanet.cuda()
        retinanet = torch.nn.DataParallel(retinanet, device_ids=[0,]).cuda()
        
    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    train()
