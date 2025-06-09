import torch
import torch.nn as nn
import numpy as np
import random
import os
import json
import argparse
from torch.utils.data import DataLoader
from datetime import datetime
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import logging
import torch
import torch.nn as nn

#import open_clip.utils.misc as misc
import open_clip
from dataset import VisaDataset, MVTecDataset
from model import linearlayer,linearlayer_att
from loss import FocalLoss, BinaryDiceLoss,BinaryFocalLoss
from prompt_ensemble import encode_text_with_prompt_ensemble
from clip_encoder import DualTokenTextEncoder
from few_shot import memory,attention_single_query


## 利用score-map计算损失，以便后续计算指标
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(args):
    # configs
    dataset_name = args.dataset
    dataset_dir = args.data_path
    epochs = args.epoch
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    image_size = args.image_size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("device: {}".format(device))
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    txt_path = os.path.join(save_path, 'log.txt')  # log

    # model configs
    features_list = args.features_list
    with open(args.config_path, 'r') as f:
        model_configs = json.load(f)

    # clip model  args.pretrained=openai   _, preprocess 就是处理数据的方式不一样
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, image_size, pretrained=args.pretrained)
    print("model: {}".format(dir(model)))
    
    #tokenizer = open_clip.get_tokenizer(args.model)

    # logger
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    root_logger.setLevel(logging.WARNING)
    logger = logging.getLogger('train')
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
        logger.info(f'{arg}: {getattr(args, arg)}')

    # transforms  这个模块主要是对图像做预处理  image size是一个超参数，应该如何去设定？
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])
    
    # datasets
    if args.dataset == 'mvtec':
        train_data = MVTecDataset(root=args.train_data_path, transform=preprocess, target_transform=transform,
                                  aug_rate=args.aug_rate)
    else:
        train_data = VisaDataset(root=args.train_data_path, transform=preprocess, target_transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    total_batches = len(train_dataloader)
    print(f"总批次数: {total_batches}")


    # losses 
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()

    dual_encoder = DualTokenTextEncoder(device,
      clip_model=model,
      num_normal_tokens=5,      # 5个正常token
      num_abnormal_tokens=5,    # 5个异常token
      freeze_base=True)

    model.to(device)
    with torch.amp.autocast(device_type='cuda'), torch.no_grad():
        
        #text_prompts = encode_text_with_prompt_ensemble(model, obj_list, tokenizer, device)
        text_prompts = dual_encoder().to(device)


    # query token 可训练的 (正态分布初始化)
    lenth = int(image_size*image_size/(model_configs['vision_cfg']['patch_size']*model_configs['vision_cfg']['patch_size']))
    query_tokens = nn.Parameter(
    torch.randn(lenth, model_configs['vision_cfg']['width']).to(device)
    )

    # linear layer
    # 对text embedding做转换，768-->1024,对齐image hidden的维度
    trainable_layer = linearlayer(model_configs['embed_dim'],model_configs['vision_cfg']['width']).to(device)
    trainable_layer_att = linearlayer_att(model_configs['vision_cfg']['width'],1).to(device)
    
    text_prompts = trainable_layer(text_prompts)
    ## 这里的可训练参数为query token，prompt token + 特征映射函数
    optimizer = torch.optim.Adam([
        {'params': trainable_layer.parameters()},
        {'params': trainable_layer_att.parameters()},
        {'params': [query_tokens, dual_encoder.normal_tokens, dual_encoder.abnormal_tokens]},
        
    ], lr=learning_rate, betas=(0.5, 0.999))

#{'params': [query_tokens, dual_encoder.normal_tokens, dual_encoder.abnormal_tokens]}
    for epoch in range(epochs):
        loss_list = []
        idx = 0
        
        for items in train_dataloader:
            idx += 1
            image = items['img'].to(device)
            cls_name = items['cls_name']  # 一个batch中对应的cls name
            with torch.amp.autocast(device_type='cuda'):
                with torch.no_grad():
                    ##features_list 代表输出哪些层的特征,这里取最后一层
                    image_features, patch_tokens_ = model.encode_image(image, features_list)
                    patch_tokens = patch_tokens_[0][:,1:,:]
                   # image_features： CLS的输出
                    #print("image_features.size: {}".format(image_features.size())) #[b,d]
                    text_features = []
                    for cls in cls_name:
                       text_features.append(text_prompts) ## 每个batch均用相同的prompt
                       #print("text_features1: {}".format(text_prompts.size())) # text_features: 
                    text_features = torch.stack(text_features, dim=0).squeeze(1) ##
                    text_features = text_features.transpose(1, 2)
                    #print("text_features: {}".format(text_features.size())) # text_features: torch.Size([8, 2, 768])

                    #text_features = text_prompts.expand(batch_size, -1, -1)
                    #print("text_features: {}".format(text_features.size())) # text_features: torch.Size([8, 2, 768])
                    # few shot 得到reference image的特征
                    #if args.mode == 'few_shot':
                    mem_features = memory(args.model, model, cls_name, dataset_dir, save_path, preprocess, transform,
                                            args.k_shot, dataset_name, device)
                    ## batch中的每一个cls_name对应的mem_features  然后根据img提取相应的特征
                    ## refer_features: [batch_size,k-shot,len,hid_dim(1024)]
                    refer_features = []
                    for cls in cls_name:
                        refer_features.append(mem_features[cls])
                    refer_features_ = torch.stack(refer_features,dim=0) ##[8,4,len,1024]
                    refer_features_ = torch.mean(refer_features_,dim=1).squeeze(1)
                    #print("refer_features: {}".format(refer_features_.size()))  ##[8,len,1024]
                    ## 进行注意力计算
                    attn_features = attention_single_query(query_tokens,refer_features_,patch_tokens[0])
                    # attn_features: [b,lenth,1024]
                attn_output = trainable_layer_att(attn_features).squeeze(2) 
                # attn_output: [b,lenth] 
                # pixel level 这里主要是训练每一层的linear 层
                patch_tokens /= patch_tokens.norm(dim=-1, keepdim=True)
                anomaly_map = (100.0 * patch_tokens @ text_features) 
                # patch_tokens [8, 1369, 768] text_features:[8,768,2]
                B, L, C = anomaly_map.shape
                H = int(np.sqrt(L)) ## 
                #print("H: {}".format(H)) 37 
                #print("image_size: {}".format(image_size)) 518
                anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                            size=image_size, mode='bilinear', align_corners=True)

                anomaly_map_att = F.interpolate(attn_output.view(B, 1,H, H),
                                            size=image_size, mode='bilinear', align_corners=True)
                ##anomaly_map_att,直接转换成B*H*H的形式            
                #print("anomaly_map: {}".format(anomaly_map.size())) [8, 2, 518, 518])
                anomaly_map = torch.softmax(anomaly_map, dim=1)

            # losses
            gt = items['img_mask'].squeeze().to(device)
            gt[gt > 0.5], gt[gt <= 0.5] = 1, 0 ## 大于0.5，为abnormal,被设置为1
            loss = 0
            loss += loss_focal(anomaly_map, gt)
            loss += loss_dice(anomaly_map[:, 1, :, :], gt) ## 第二维度 为abnormal的概率
            loss += loss_dice(anomaly_map_att[:, 0, :, :], gt) ## 第二维度 为abnormal的概率

            ## attn_features: [b,lenth,1] 匹配输出，为1的概率
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

        # logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))

        # save model
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(save_path, 'epoch_' + str(epoch + 1) + '.pth')
            torch.save({'trainable_linearlayer': trainable_layer.state_dict()}, ckp_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser("VAND Challenge", add_help=True)
    # path
    parser.add_argument("--data_path", type=str, default="./data/visa", help="path to test dataset")
    parser.add_argument("--train_data_path", type=str, default="./data/visa", help="train dataset path")
    parser.add_argument("--save_path", type=str, default='./exps/vit_large_14_518', help='path to save results')
    parser.add_argument("--config_path", type=str, default='./open_clip/model_configs/ViT-B-16.json', help="model configs")
    # model
    parser.add_argument("--dataset", type=str, default='mvtec', help="train dataset name")
    parser.add_argument("--model", type=str, default="ViT-B-16", help="model used")
    parser.add_argument("--pretrained", type=str, default="laion400m_e32", help="pretrained weight used")
    parser.add_argument("--features_list", type=int, nargs="+", default=[3, 6, 9], help="features used")
    # hyper-parameter
    parser.add_argument("--epoch", type=int, default=200, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--image_size", type=int, default=224, help="image size")
    parser.add_argument("--aug_rate", type=float, default=0.2, help="image size")
    parser.add_argument("--print_freq", type=int, default=30, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=3, help="save frequency")
    parser.add_argument("--k_shot", type=int, default=10, help="e.g., 10-shot, 5-shot, 1-shot")

    args = parser.parse_args()

    setup_seed(111)
    train(args)