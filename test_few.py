import os
import cv2
import json
import torch
import random
import logging
import argparse
import numpy as np
from PIL import Image
from skimage import measure
from tabulate import tabulate
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.metrics import auc, roc_auc_score, average_precision_score, f1_score, precision_recall_curve, pairwise
from tqdm import tqdm

import open_clip
from few_shot import memory,attention
from model import Classifier
from dataset import VisaDataset, MVTecDataset
from prompt_ensemble import encode_text_with_prompt_ensemble
from clip_encoder_test import DualTokenTextEncoder
from metric import image_level_metrics

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)

def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

def cal_pro_score(masks, amaps, max_step=200, expect_fpr=0.3):
    # ref: https://github.com/gudovskiy/cflow-ad/blob/master/train.py
    binary_amaps = np.zeros_like(amaps, dtype=bool)
    min_th, max_th = amaps.min(), amaps.max()
    delta = (max_th - min_th) / max_step
    pros, fprs, ths = [], [], []
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th], binary_amaps[amaps > th] = 0, 1
        pro = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                tp_pixels = binary_amap[region.coords[:, 0], region.coords[:, 1]].sum()
                pro.append(tp_pixels / region.area)
        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()
        pros.append(np.array(pro).mean())
        fprs.append(fpr)
        ths.append(th)
    pros, fprs, ths = np.array(pros), np.array(fprs), np.array(ths)
    idxes = fprs < expect_fpr
    fprs = fprs[idxes]
    fprs = (fprs - fprs.min()) / (fprs.max() - fprs.min())
    pro_auc = auc(fprs, pros[idxes])
    return pro_auc


def test(args):
    img_size = args.image_size
    dataset_dir = args.data_path
    save_path = args.save_path
    dataset_name = args.dataset
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    txt_path = os.path.join(save_path, 'log_test_few_map.txt')

    # clip
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, img_size, pretrained=args.pretrained)
    model.to(device)

    # logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('test')
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    logger.setLevel(logging.INFO)
    file_handler = logging.FileHandler(txt_path, mode='w')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # record parameters
    for arg in vars(args):
        if args.mode == 'zero_shot' and (arg == 'k_shot' or arg == 'few_shot_features'):
            continue
        logger.info(f'{arg}: {getattr(args, arg)}')

    dual_encoder = DualTokenTextEncoder(device,
      clip_model=model,
      num_normal_tokens=5,      # 5个正常token
      num_abnormal_tokens=5,    # 5个异常token
      freeze_base=True)


    model.to(device)

    # seg
    with open(args.config_path, 'r') as f:
        model_configs = json.load(f)

    trainable_layer = Classifier(model_configs['embed_dim'],2).to(device)
    ## 参数加载
    checkpoint = torch.load(args.checkpoint_path)
    dual_encoder.load_state_dict(checkpoint["dual_encoder"])
    query_tokens=checkpoint["query_tokens"].to(device)
    trainable_layer.load_state_dict(checkpoint["trainable_layer"])
    
    print("query_tokens: {}".format(query_tokens.size()))
    # dataset
    transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor()
        ])

    if dataset_name == 'mvtec':
        test_data = MVTecDataset(root=dataset_dir, transform=preprocess, target_transform=transform,
                                 aug_rate=-1, mode='test')
    else:
        test_data = VisaDataset(root=dataset_dir, transform=preprocess, target_transform=transform, mode='test')
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)
    obj_list = test_data.get_cls_names() 

    # few shot
    ## get k-shot features of all cls names.
    if args.mode == 'few_shot':    
        mem_features = memory(args.model, model, obj_list, dataset_dir, save_path, preprocess, transform,
                                            args.k_shot, dataset_name, device)
    results = {}
    metrics = {}
    for obj in obj_list:
        results[obj] = {}
        results[obj]['gt_sp'] = []
        results[obj]['pr_sp'] = []
        metrics[obj] = {}
        metrics[obj]['image-auroc'] = 0
        metrics[obj]['image-ap'] = 0
    with torch.amp.autocast(device_type=device), torch.no_grad():
        text_features = dual_encoder() # [1, 2, dim]
    text_features = text_features/text_features.norm(dim=-1, keepdim=True)
    for idx, items in enumerate(tqdm(test_dataloader)):
        image = items['img'].to(device)
        cls_name = items['cls_name']
        results[cls_name[0]]['gt_sp'].extend(items['anomaly'].detach().cpu())
        with torch.no_grad():
            image_features, _ = model.encode_image(image, [24])
            image_features = image_features / image_features.norm(dim=-1, keepdim=True) #[1, dim]

            text_probs = image_features.unsqueeze(1).float() @ text_features.permute(0, 2, 1).float()
            ## refer-score
            refer_feature= mem_features[cls_name[0]] #[k-shot, dim]
            refer_feature_ = torch.mean(refer_feature,dim=0,keepdim=True)  #[1,dim]
            attn_features = attention(query_tokens,refer_feature_,image_features)
            refer_score = trainable_layer(attn_features)

            f_probs = text_probs.squeeze(1) + refer_score
            f_probs = f_probs.softmax(-1)
            f_probs = f_probs[:, 1] ####取为abnormal的概率

            results[cls_name[0]]['pr_sp'].extend(f_probs.detach().cpu())

    table_ls = []
    image_auroc_list = []
    image_ap_list = []

    for obj in obj_list:
        table = []
        table.append(obj)
        if args.metrics == 'image-level':
            image_auroc = image_level_metrics(results, obj, "image-auroc")
            image_ap = image_level_metrics(results, obj, "image-ap")
            table.append(str(np.round(image_auroc * 100, decimals=1)))
            table.append(str(np.round(image_ap * 100, decimals=1)))
            image_auroc_list.append(image_auroc)
            image_ap_list.append(image_ap)
        table_ls.append(table) 
        # logger
    table_ls.append(['mean', 
                        str(np.round(np.mean(image_auroc_list) * 100, decimals=1)),
                        str(np.round(np.mean(image_ap_list) * 100, decimals=1))])
    results_ = tabulate(table_ls, headers=['objects', 'image_auroc', 'image_ap'], tablefmt="pipe")
    logger.info("\n%s", results_)
if __name__ == '__main__':
    parser = argparse.ArgumentParser("VAND Challenge", add_help=True)
    # paths
    parser.add_argument("--data_path", type=str, default="./data/visa", help="path to test dataset")
    parser.add_argument("--save_path", type=str, default='./results/tiaoshi', help='path to save results')
    parser.add_argument("--checkpoint_path", type=str, default='./exps/vit_huge_14/model_epoch12.pth', help='path to save results')
    parser.add_argument("--config_path", type=str, default='./open_clip/model_configs/ViT-B-16.json', help="model configs")
    # model
    parser.add_argument("--dataset", type=str, default='mvtec', help="test dataset")
    parser.add_argument("--model", type=str, default="ViT-B-16", help="model used")
    parser.add_argument("--pretrained", type=str, default="laion400m_e32", help="pretrained weight used")
    parser.add_argument("--features_list", type=int, nargs="+", default=[3, 6, 9], help="features used")
    parser.add_argument("--few_shot_features", type=int, nargs="+", default=[3, 6, 9], help="features used for few shot")
    parser.add_argument("--image_size", type=int, default=224, help="image size")
    parser.add_argument("--mode", type=str, default="zero_shot", help="zero shot or few shot")
    # few shot
    parser.add_argument("--metrics", type=str, default='image-level')
    parser.add_argument("--k_shot", type=int, default=10, help="e.g., 10-shot, 5-shot, 1-shot")
    parser.add_argument("--seed", type=int, default=10, help="random seed")
    args = parser.parse_args()

    setup_seed(args.seed)
    test(args)

