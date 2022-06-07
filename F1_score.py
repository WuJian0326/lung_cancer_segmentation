import numpy as np
from SEG_Train_Datasets.Image_Process import get_all_file_extension_path
import cv2
from tqdm.contrib import tzip
from PIL import Image

def F1_score(premask, groundtruth):
    # PT PF NT NF
    seg_inv, gt_inv = np.logical_not(premask), np.logical_not(groundtruth)
    true_pos = float(np.logical_and(premask, groundtruth).sum())  # float for division
    true_neg = np.logical_and(seg_inv, gt_inv).sum()
    false_pos = np.logical_and(premask, gt_inv).sum()
    false_neg = np.logical_and(seg_inv, groundtruth).sum()

    # 計算F1 socre
    prec = true_pos / (true_pos + false_pos + 1e-6)
    rec = true_pos / (true_pos + false_neg + 1e-6)
    F1 = 2 * prec * rec / (prec + rec)
    # accuracy = (true_pos + true_neg) / (true_pos + true_neg + false_pos + false_neg + 1e-6)
    # F1 = 2 * true_pos / (2 * true_pos + false_pos + false_neg + 1e-6)
    # IoU = true_pos / (true_pos + false_neg + false_pos + 1e-6)
    return F1


mask = 'SEG_Train_Datasets/Test/output_mask'
truth = 'SEG_Train_Datasets/Test/traintarget'

mask_path = get_all_file_extension_path(mask, 'png')
truth_path = get_all_file_extension_path(truth, 'jpg')

F1 = 0
for idx, (m, t) in enumerate(tzip(mask_path, truth_path)):
    premask = np.array(Image.open(m))
    groundturth = np.array(Image.open(t))
    F1 = F1 + F1_score(premask, groundturth)

print('F1_score', F1/len(mask_path))