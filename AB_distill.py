import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import argparse
import collections

import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms

from retinanet import model
from distillation import AB_distillation_module
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

from retinanet import coco_eval

import csv


def distill():
    print("Start distillation")

    if not os.path.exists('logs/50_18_distill'):
        os.makedirs('logs/50_18_distill')
    file_distill = open('logs/50_18_distill/distill.csv', 'w')
    distill_writer = csv.writer(file_distill)

    for epoch_num in range(parser.distill_epochs):
        distillation_module.train()
        distillation_module.s_net.train()
        distillation_module.t_net.eval()
        distillation_module.s_net.freeze_bn()
        distillation_module.t_net.freeze_bn()
            

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()

                inputs = None
                if use_gpu and torch.cuda.is_available():
                    inputs = data['img'].cuda().float()
                else:
                    inputs = data['img'].float()

                distillation_module.batch_size = inputs.shape[0]

                loss, loss_AT4, loss_AT3, loss_AT2, loss_AT1 = distillation_module(inputs)

                loss.backward()
                optimizer.step()

                print(
                    'Epoch: {} | Iteration: {} | Loss: {:1.5f} | Loss AT1: {:1.5f} | Loss AT2: {:1.5f} | Loss AT3: {:1.5f} | Loss AT4: {:1.5f}'.format(
                        epoch_num, iter_num, loss.item(),  loss_AT1.item(), loss_AT2.item(), loss_AT3.item(), loss_AT4.item()))

                distill_writer.writerow([epoch_num, iter_num, loss.item(),  loss_AT1.item(), loss_AT2.item(), loss_AT3.item(), loss_AT4.item()])
                file_distill.flush()

                del loss
                del loss_AT4
                del loss_AT3
                del loss_AT2
                del loss_AT1
            except Exception as e:
                print(e)
                continue

        if not os.path.exists("models/50_18_distill"):
            os.makedirs("models/50_18_distill")

        torch.save(student, "models/{}/{}_retinanet{}_{}.pt".format("50_18_distill", parser.dataset, "50_18_distill", epoch_num))


def train():
    print("Start training")

    if not os.path.exists('logs/50_18_distill'):
        os.makedirs('logs/50_18_distill')
    file_train = open('logs/50_18_distill/train.csv', 'w')
    file_val = open('logs/50_18_distill/val.csv', 'w')
    train_writer = csv.writer(file_train)
    val_writer = csv.writer(file_val)

    loss_hist = collections.deque(maxlen=500)

    for epoch_num in range(parser.train_epochs):
        student.train()
        student.freeze_bn()

        epoch_loss = []

        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()

                if use_gpu and torch.cuda.is_available():
                    classification_loss, regression_loss = student([data['img'].float().cuda(), data['annot'].cuda()])
                else:
                    classification_loss, regression_loss = student([data['img'].float(), data['annot']])
                    
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(student.parameters(), 0.1)

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

        if not os.path.exists("models/50_18_train"):
            os.makedirs("models/50_18_train")

        torch.save(student, 'models/{}/{}_retinanet{}_{}.pt'.format("50_18_train", parser.dataset, "50_18_train", epoch_num))

        scheduler.step(np.mean(epoch_loss))

        if parser.dataset == 'coco':
            print('Evaluating dataset')
            stats = coco_eval.evaluate_coco(dataset_val, student)
            val_writer.writerow(stats)
            file_val.flush()

    student.eval()
    
    file_train.close()
    file_val.close()


if __name__ == "__main__":
    # parse args
    parser = argparse.ArgumentParser(description='Simple distillation script for RetinaNet with different backends.')
    
    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory.')
    parser.add_argument('--teacher_path', help='Path to teacher model.')

    parser.add_argument('--distill_epochs', help='Number of epochs for distillation.', type=int, default=10)
    parser.add_argument('--train_epochs', help='Number of epochs for training.', type=int, default=100)
    parser.add_argument('--batch_size', help='Size of batch.', type=int, default=2)

    parser = parser.parse_args()

    # create data loaders
    if parser.dataset == 'coco':
        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')
        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':
        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')
        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')
        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=parser.batch_size, drop_last=False)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=parser.batch_size, drop_last=False)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    
    use_gpu = True

    # initialize models
    teacher = torch.load(parser.teacher_path, map_location=lambda storage, loc: storage)
    student = model.resnet18(num_classes=dataset_train.num_classes(), pretrained=False, use_gpu=use_gpu)
    distillation_module = AB_distillation_module.AB_Distill_Resnet_50_18(teacher, student, None, 1)

    if use_gpu and torch.cuda.is_available():
        teacher.cuda()
        student.cuda()
        distillation_module.cuda()
        teacher.switch_to_gpu()
        
        # teacher = torch.nn.DataParallel(teacher, device_ids=[0,]).cuda()
        # student = torch.nn.DataParallel(student, device_ids=[0,]).cuda()
        # distillation_module = torch.nn.DataParallel(distillation_module, device_ids=[0,]).cuda()
    else:
        teacher.switch_to_cpu()

    teacher.training = True
    student.training = True

    # optimizer for distillation
    # optimizer = optim.SGD([{'params': student.parameters()},
    #                    {'params': distillation_module.Connectors.parameters()}], lr=0.001, nesterov=True, momentum=0.9, weight_decay=1e-4)
    optimizer = optim.Adam([{'params': student.parameters()},
                        {'params': distillation_module.Connectors.parameters()}], lr=1e-3)

    distill()

    # optimizer for training
    optimizer = optim.Adam(student.parameters(), lr=1e-5)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    train()
