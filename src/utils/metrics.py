import torch
import numpy as np
from sklearn.metrics import *

def one_hot_encoder(n_classes, input_tensor):
    tensor_list = []
    for i in range(n_classes):
        temp_prob = input_tensor == i * torch.ones_like(input_tensor)
        tensor_list.append(temp_prob)
    output_tensor = torch.cat(tensor_list, dim=1)
    return output_tensor.float()

def get_accuracy(SR, GT, threshold=0.5):
    SR = SR > threshold
    GT = GT == torch.max(GT)
    corr = torch.sum(SR == GT)
    tensor_size = SR.size(0)*SR.size(1)
    acc = float(corr)/float(tensor_size)
    return acc


def get_sensitivity(SR, GT, threshold=0.5):
    # Sensitivity == Recall
    SE = 0
    SR = SR > threshold
    GT = GT == torch.max(GT)
    # TP : True Positive
    # FN : False Negative
    TP = ((SR == 1).byte() + (GT == 1).byte()) == 2
    FN = ((SR == 0).byte() + (GT == 1).byte()) == 2
    SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)
    return SE

def get_specificity(SR,GT,threshold=0.5):
    SP = 0
    SR = SR > threshold
    GT = GT == torch.max(GT)
    # TN : True Negative
    # FP : False Positive
    TN = ((SR == 0).byte() + (GT == 0).byte()) == 2
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2
    SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
    return SP

def get_precision(SR,GT,threshold=0.5):
    PC = 0
    SR = SR > threshold
    GT = GT== torch.max(GT)
    # TP : True Positive
    # FP : False Positive
    TP = ((SR == 1).byte() + (GT == 1).byte()) == 2
    FP = ((SR == 1).byte() + (GT == 0).byte()) == 2
    PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)
    return PC

def iou_score(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy()
    output_ = output > 0.5
    target_ = target > 0.5

    if target_.sum() == 0 and output_.sum() == 0:
        return 1,1,1,1,1,1,1

    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()
    iou = intersection / (union + smooth)
    dice = (2 * iou) / (iou + 1)

    
    output_ = torch.tensor(output_)
    target_ = torch.tensor(target_)
    SE = get_sensitivity(output_, target_, threshold=0.5)
    PC = get_precision(output_, target_, threshold=0.5)
    SP= get_specificity(output_, target_, threshold=0.5)
    ACC=get_accuracy(output_, target_, threshold=0.5)
    F1 = 2*SE*PC/(SE+PC + 1e-6)


    return iou, dice, SE, PC, F1, SP, ACC


def iou_score2(output, target):
    smooth = 1e-5

    if torch.is_tensor(output):
        output = torch.sigmoid(output).data.cpu().numpy()
    if torch.is_tensor(target):
        target = target.data.cpu().numpy().astype(int)
    output_ = (output > 0.5).astype(int)
    target_ = target

    # if target_.sum(axis=(1, 2)) == 0 and output_.sum() == 0:
    #     return 1, 1, 1, 1, 1, 1


    intersection = np.sum(output_ * target_, axis=(1, 2))
    dice = (2 * intersection) / (np.sum(target_, axis=(1,2)) + np.sum(output_, axis=(1,2)) + smooth)

    intersection = np.logical_and(output_, target_)
    union = np.logical_or(output_, target_)
    iou = np.sum(intersection,axis=(1,2)) / (np.sum(union , axis=(1,2))+smooth)

    dice = np.where(np.sum(target_, axis=(1, 2)) == 0 and np.sum(output_, axis=(1, 2)) == 0, 1, dice)
    iou = np.where(np.sum(target_, axis=(1, 2)) == 0 and np.sum(output_, axis=(1, 2)) == 0, 1, iou)


    output_flat = output_.reshape(target_.shape[0], -1)
    target_flat = target_.reshape(target_.shape[0], -1)


    PC = []
    SE =[]
    F1 = []
    for i in range(target_.shape[0]):
        PC_t, SE_t, F1_t, _ = precision_recall_fscore_support(target_flat[i], output_flat[i], average="binary")
        PC.append(PC_t)
        SE.append(SE_t)
        F1.append(F1_t)

    # ACC = accuracy_score(target_flat[0], output_flat[0])

    return np.mean(iou), np.mean(dice), np.mean(SE), np.mean(PC), np.mean(F1)

def get_prec_recall(preds, labels):
    # assert (preds.numel() == labels.numel())
    preds_ = preds
    labels_ = labels
    # preds_ = preds_.squeeze(1)
    # labels_ = labels_.squeeze(1)
    assert (len(preds_.shape) == len(labels_.shape))
    assert (len(preds_.shape) == 3)
    prec_list = []
    recall_list = []

    assert (preds_.min() >= 0 and preds_.max() <= 1)
    assert (labels_.min() >= 0 and labels_.max() <= 1)
    for i in range(preds_.shape[0]):
        pred_, label_ = preds_[i], labels_[i]
        thres_ = pred_.sum() * 2.0 / pred_.size

        binari_ = np.zeros(shape=pred_.shape, dtype=np.uint8)
        binari_[np.where(pred_ >= 0.5)] = 1

        label_ = label_.astype(np.uint8)
        matched_ = np.multiply(binari_, label_)

        TP = matched_.sum()
        TP_FP = binari_.sum()
        TP_FN = label_.sum()
        prec = (TP + 1e-6) / (TP_FP + 1e-6)
        recall = (TP + 1e-6) / (TP_FN + 1e-6)
        prec_list.append(prec)
        recall_list.append(recall)
    return prec_list, recall_list

def dice_coef(output, target):
    smooth = 1e-5
    output = torch.sigmoid(output).view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()
    return (2. * intersection + smooth) / \
        (output.sum() + target.sum() + smooth)
        

