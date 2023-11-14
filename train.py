import argparse
import datetime
import random
import time
from torch.optim import lr_scheduler
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torch.nn.parallel import DataParallel
from torchvision import transforms
from albumentations.augmentations import transforms
import albumentations as albu
from albumentations.core.composition import Compose
from albumentations import RandomRotate90, Resize
import src.utils.losses as losses
from src.utils.util import AverageMeter
from src.utils.metrics import *
from src.utils import ramps
from src.dataloader.dataset import SemiDataSets, SemiDataSets_sing, TwoStreamBatchSampler
from src.network.MGCC import MGCC
import os
import csv
import cv2
# os.environ["CUDA_VISIBLE_DEVICES"] = '2,0'

def seed_torch(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


parser = argparse.ArgumentParser()
parser.add_argument('--ex_name', type=str, default="1") # 14, 38, 75, 113  us  76 149
parser.add_argument('--class_num', type=int, default=3) # only roi
parser.add_argument('--semi_percent', type=float, default=0.5)
parser.add_argument('--dataset_name', type=str, default="camus")
parser.add_argument('--base_dir', type=str, default="./data/", help='dir')
parser.add_argument('--train_file_dir', type=str, default="camus_train3.txt", help='dir')
parser.add_argument('--val_file_dir', type=str, default="camus_val3.txt", help='dir')
parser.add_argument('--size', type=int,  default=256, help='size of network input')
parser.add_argument('--max_iterations', type=int,
                    default=40000, help='maximum epoch number to train')
parser.add_argument('--total_batch_size', type=int, default=8,
                    help='batch_size per gpu')
parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')
parser.add_argument('--min_lr', default=1e-5, type=float,
                    help='minimum learning rate')
parser.add_argument('--seed', type=int, default=456, help='random seed')
# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=4,
                    help='labeled_batch_size per gpu')
# loss
parser.add_argument('--loss', type=str, default='BCEDiceLoss')
# costs
parser.add_argument('--consistency', type=float,
                    default=7, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')
# MGCC hyperparameter
parser.add_argument('--kernel_size', type=int,
                    default=7, help='ConvMixer kernel size')
parser.add_argument('--length', type=tuple,
                    default=(3, 3, 3), help='length of ConvMixer')
args = parser.parse_args()

seed_torch(args.seed)


def getDataloader(args):
    train_transform = Compose([
        RandomRotate90(),
        albu.Flip(),
        Resize(args.size, args.size),
        transforms.Normalize(),
    ])
    val_transform = Compose([
        Resize(args.size, args.size),
        transforms.Normalize(),
    ])
    labeled_slice = args.semi_percent


    # db_train = SemiDataSets_sing(args = args, base_dir=args.base_dir, split="train", transform=train_transform,
    #                         train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir,
    #                         )
    # db_val = SemiDataSets_sing(args = args, base_dir=args.base_dir, split="val", transform=train_transform,
    #                       train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir
    #                       )


    db_train = SemiDataSets(args = args, base_dir=args.base_dir, split="train", transform=train_transform,
                            train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir,
                            )
    db_val = SemiDataSets(args = args, base_dir=args.base_dir, split="val", transform=val_transform,
                          train_file_dir=args.train_file_dir, val_file_dir=args.val_file_dir
                          )

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    total_slices = len(db_train)
    labeled_idxs = list(range(0, int(labeled_slice * total_slices)))
    unlabeled_idxs = list(range(int(labeled_slice * total_slices), total_slices))
    print("label num:{}, unlabel num:{} percent:{}".format(len(labeled_idxs), len(unlabeled_idxs), labeled_slice))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.total_batch_size, args.labeled_bs)
    trainloader = DataLoader(db_train, batch_sampler=batch_sampler,
                             num_workers=8, pin_memory=True, worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=args.total_batch_size, shuffle=False, num_workers=8)

    return trainloader, valloader


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def getModel(args):
    print("ConvMixer1:{}, ConvMixer2:{}, ConvMixer3:{}, kernal:{}".format(args.length[0], args.length[1],
                                                                          args.length[2], args.kernel_size))
    return MGCC(args = args, length=args.length, k=args.kernel_size).cuda()


def train(args):
    config = vars(args)
    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    base_lr = args.base_lr
    max_iterations = int(args.max_iterations * args.semi_percent)
    trainloader, valloader = getDataloader(args)

    model = getModel(args).cuda()
    model = DataParallel(model)
    print("{} iterations per epoch".format(len(trainloader)))
    best_iou = 0
    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    max_epoch = 300

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    # optimizer = optim.Adam(model.parameters(), lr=base_lr,  weight_decay=0.0001)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epoch, eta_min=config['min_lr'])
    criterion = losses.__dict__[config['loss']]().cuda()
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(config['class_num'])
    for epoch_num in range(max_epoch):
        avg_meters = {'total_loss': AverageMeter(),
                      'train_iou': AverageMeter(),
                      'train_dice': AverageMeter(),
                      'consistency_loss': AverageMeter(),
                      'supervised_loss': AverageMeter(),
                      'val_loss': AverageMeter(),
                      'val_iou': AverageMeter(),
                      'val_dice': AverageMeter(),
                      'val_se': AverageMeter(),
                      'val_pc': AverageMeter(),
                      'val_f1': AverageMeter(),
                      'val_acc': AverageMeter(),
                      }
        for i in range(args.class_num):
            for m in ["dice", "iou", "se", "pc", "f1", "acc"]:
                avg_meters[f'{i}_{m}'] = AverageMeter()


        model.train()
        for i_batch, sampled_batch in enumerate(trainloader):

            name = sampled_batch['name']
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            label_batch_onehot = one_hot_encoder(args.class_num, label_batch)

            outputs, outputs_aux1, outputs_aux2, outputs_aux3 = model(volume_batch)

            outputs_soft = torch.sigmoid(outputs)
            outputs_aux1_soft = torch.sigmoid(outputs_aux1)
            outputs_aux2_soft = torch.sigmoid(outputs_aux2)
            outputs_aux3_soft = torch.sigmoid(outputs_aux3)


            bs_loss = 0
            all_iou, all_dice = [], []

            for i in range(args.class_num):

                loss_ce = criterion(outputs[:args.labeled_bs, i],
                                    label_batch[:args.labeled_bs, i])
                loss_ce_aux1 = criterion(outputs_aux1[:args.labeled_bs, i],
                                         label_batch[:args.labeled_bs, i])
                loss_ce_aux2 = criterion(outputs_aux2[:args.labeled_bs, i],
                                         label_batch[:args.labeled_bs, i])
                loss_ce_aux3 = criterion(outputs_aux3[:args.labeled_bs, i],
                                         label_batch[:args.labeled_bs, i])

                supervised_loss = (loss_ce + loss_ce_aux1 + loss_ce_aux2 + loss_ce_aux3) / 4


                consistency_weight = get_current_consistency_weight(iter_num // 150)
                consistency_loss_aux1 = torch.mean(
                    (outputs_soft[args.labeled_bs:, i] - outputs_aux1_soft[args.labeled_bs:, i]) ** 2)
                consistency_loss_aux2 = torch.mean(
                    (outputs_soft[args.labeled_bs:, i] - outputs_aux2_soft[args.labeled_bs:, i]) ** 2)
                consistency_loss_aux3 = torch.mean(
                    (outputs_soft[args.labeled_bs:, i] - outputs_aux3_soft[args.labeled_bs:, i]) ** 2)

                consistency_loss = (consistency_loss_aux1 + consistency_loss_aux2 + consistency_loss_aux3) / 3
                loss = supervised_loss + consistency_weight * consistency_loss
                bs_loss += loss

                iou, dice, _, _, _, _, _ = iou_score(outputs[:args.labeled_bs, i], label_batch[:args.labeled_bs, i])
                all_iou.append(iou)
                all_dice.append(dice)

            bs_loss = bs_loss / args.class_num
            # print(bs_loss)

            optimizer.zero_grad()
            bs_loss.backward()
            optimizer.step()
            scheduler.step()


            iter_num = iter_num + 1

            avg_meters['total_loss'].update(bs_loss.item(), volume_batch[:args.labeled_bs].size(0))
            avg_meters['supervised_loss'].update(supervised_loss.item(), volume_batch[:args.labeled_bs].size(0))
            avg_meters['consistency_loss'].update(consistency_loss.item(), volume_batch[args.labeled_bs:].size(0))
            avg_meters['train_iou'].update(np.mean(all_iou), volume_batch[:args.labeled_bs].size(0))
            avg_meters['train_dice'].update(np.mean(all_dice), volume_batch[:args.labeled_bs].size(0))

        del volume_batch, label_batch ,outputs, outputs_aux1, outputs_aux2, outputs_aux3
        print(
            'train epoch [%3d/%d]  %s  train_loss %.4f supervised_loss %.4f consistency_loss %.4f train_iou: %.4f  train_dice %.4f '
            % (epoch_num, max_epoch, datetime.datetime.now(), avg_meters['total_loss'].avg,
               avg_meters['supervised_loss'].avg, avg_meters['consistency_loss'].avg, avg_meters['train_iou'].avg,
               avg_meters['train_dice'].avg))

        if epoch_num % 5 == 0:
            model.eval()

            with torch.no_grad():
                for i_batch, sampled_batch in enumerate(valloader):
                    input, target = sampled_batch['image'], sampled_batch['label']
                    name = sampled_batch['name']
                    input = input.cuda()
                    target = target.cuda()

                    target_onehot = one_hot_encoder(args.class_num, target)

                    output = model(input)

                    all_loss = 0
                    all_iou, all_dice, all_SE, all_PC, all_F1, all_ACC = [], [], [], [], [], []


                    for i in range(args.class_num):

                        loss = criterion(output[:, i], target_onehot[:, i])
                        all_loss += loss

                        metric_list = 0.0
                        for xx in range(input.size(0)):
                            metric_i = iou_score(output[xx, i], target_onehot[xx, i])
                            metric_list += np.array(metric_i)
                        metric_list = metric_list / input.size(0)

                        # print(metric_list, "**************")
                        # iou, dice, SE, PC, F1, = iou_score2(output[:, i], target[:, i])
                        # print(iou, dice, SE, PC, F1, "@@@@@@@@@@@@@@@", input.size(0))

                        avg_meters[f'{i}_iou'].update(metric_list[0], input.size(0))
                        avg_meters[f'{i}_dice'].update(metric_list[1], input.size(0))
                        avg_meters[f'{i}_se'].update(metric_list[2], input.size(0))
                        avg_meters[f'{i}_pc'].update(metric_list[3], input.size(0))
                        avg_meters[f'{i}_f1'].update(metric_list[4], input.size(0))
                        avg_meters[f'{i}_acc'].update(metric_list[6], input.size(0))

                        all_iou.append(metric_list[0])
                        all_dice.append(metric_list[1])
                        all_SE.append(metric_list[2])
                        all_PC.append(metric_list[3])
                        all_F1.append(metric_list[4])
                        all_ACC.append(metric_list[6])


                    all_loss = all_loss / args.class_num

                    avg_meters['val_loss'].update(all_loss.item(), input.size(0))
                    avg_meters['val_iou'].update(np.mean(all_iou), input.size(0))
                    avg_meters['val_dice'].update(np.mean(all_dice), input.size(0))
                    avg_meters['val_se'].update(np.mean(all_SE), input.size(0))
                    avg_meters['val_pc'].update(np.mean(all_PC), input.size(0))
                    avg_meters['val_f1'].update(np.mean(all_F1), input.size(0))
                    avg_meters['val_acc'].update(np.mean(all_ACC), input.size(0))

            print(
                'test epoch [%3d/%d]  %s   val_loss %.4f - val_iou %.4f - val_dice %.4f - val_SE %.4f - val_PC %.4f - val_F1 %.4f - val_ACC %.4f'
                % (epoch_num, max_epoch, datetime.datetime.now(),
                   avg_meters['val_loss'].avg, avg_meters['val_iou'].avg, avg_meters['val_dice'].avg, avg_meters['val_se'].avg,
                   avg_meters['val_pc'].avg, avg_meters['val_f1'].avg, avg_meters['val_acc'].avg,))

            print([f"{i}_{m}: {avg_meters[str(i) + '_' + m].avg}" for i in range(args.class_num)
                                                                    for m in ["dice", "iou", "se", "pc", "f1", "acc"]])
            if avg_meters['val_iou'].avg > best_iou:
                torch.save(model.state_dict(), f'checkpoint/model_{args.dataset_name}.pth')
                best_iou = avg_meters['val_iou'].avg
                print("=> saved best model   @@@@@@@@@@@@   ")

    return "Training Finished!"


if __name__ == "__main__":
    train(args)
