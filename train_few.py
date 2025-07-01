import torch
import torch.nn as nn
import numpy as np
import random
import os
import json
import argparse
import wandb

from torch.utils.data import DataLoader
from datetime import datetime
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import logging
import torch
import torch.nn as nn
from loss import FocalLoss, BinaryDiceLoss

#import open_clip.utils.misc as misc
import open_clip
from dataset import VisaDataset, MVTecDataset
from loss import FocalLoss, BinaryDiceLoss
from prompt_ensemble import FewVand_PromptLearner
from few_shot import memory_pixel,patch_attention



# 初始化项目
#wandb.init(project="few-VAND-loss")

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir='./runs')  # 创建日志目录

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
    n_ctx = args.n_ctx
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
    model.eval()
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
        ### 为什么这里的mode默认为test
        train_data = MVTecDataset(root=args.train_data_path, transform=preprocess, target_transform=transform,
                                  aug_rate=args.aug_rate)
    else:
        train_data = VisaDataset(root=args.train_data_path, transform=preprocess, target_transform=transform)
    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    total_batches = len(train_dataloader)
    print(f"总批次数: {total_batches}")

    image_size_config = image_size
    patch_size_config = model_configs["vision_cfg"]['patch_size']
    num_query_tokens = int((image_size_config/patch_size_config)**2)
    width = model_configs["vision_cfg"]['width']
    emb_dim = model_configs['embed_dim']

    Ppra = {"num_query_tokens":num_query_tokens,"width":width,"emb_dim":emb_dim}
    # learnable prompt  
    with torch.cuda.amp.autocast():
        prompt_learner = FewVand_PromptLearner(model.to(device),n_ctx,device,Ppra)
    
    prompt, tokenized_prompt = prompt_learner()

    #print("text_features: {}".format(text_features.size())) # [768, 2]
    model.to(device)
    prompt_learner.to(device)

    #print("prompt_learner:",prompt_learner.ctx_neg)
    # query token 可训练的 (正态分布初始化)
    #num_query_tokens = 1
    # query_tokens = nn.Parameter(
    # torch.randn(num_query_tokens, model_configs['embed_dim']).to(device)
    # ) image-level

# pixel-level

    # linear layer
    ## 对attention 进行线性变换，输出二分类结果
    #trainable_layer = Classifier(model_configs['embed_dim'],2).to(device)
    query_tokens = prompt_learner.query_token
    ## 这里的可训练参数为query token，prompt token + 最后的分类器
    optimizer = torch.optim.Adam(list(prompt_learner.parameters()), lr=learning_rate, betas=(0.5, 0.999))

    model.eval()
    prompt_learner.train()
    # 打印可训练的参数
    for name, param in prompt_learner.named_parameters():
        #print(name)
        if param.requires_grad:  # 仅打印可训练的参数
            print(f"Parameter Name: {name}, Parameter Value: {param.data.size()}")

    # losses
    loss_focal = FocalLoss()
    loss_dice = BinaryDiceLoss()

    # 所有类别均共享相同的normal和abnormal prompt
    #torch.autograd.set_detect_anomaly(True)  # 启用异常检测
    
    for epoch in range(epochs):
        loss_list = []
        idxs = 0
        for items in train_dataloader:
            idxs +=1
            image = items['img'].to(device)
            cls_name = items['cls_name']  # 一个batch中对应的cls name
            with torch.amp.autocast(device_type="cuda"):
                text_features = model.encode_text_learn(prompt, tokenized_prompt)
                text_features = text_features.permute(1, 0) # # [768, 2]
                with torch.no_grad():
                    ## 取最后一层的patch embedding
                    image_features, patch_features = model.encode_image(image, features_list)
                    print("patch_features: ",len(patch_features),patch_features[0].size())
                    # few shot 得到reference image的特征
                    mem_features = memory_pixel(args.model, model, cls_name, dataset_dir, save_path, preprocess, transform,
                                            args.k_shot, dataset_name, device)
                    ## refer_features: [batch_size,k-shot,emb_dim]
                    refer_features = []
                    for cls in cls_name:
                        refer_features.append(mem_features[cls])
            
                    refer_features = torch.stack(refer_features,dim=0) ##[batch,num_patch,emb_dim]
                    ## 交叉注意力计算
                    #attn_features = attention(query_tokens,refer_features_,image_features) image-level
                if 'ViT' in args.model:
                    patch_features = [p[:, 1:, :] for p in patch_features] # 第一个位置为cls
                print("q,k,v: ",query_tokens.size(),refer_features.size(),patch_features[0].size()) 
                # [batch,patch len, hidden dim]
                attn_features = patch_attention(query_tokens,refer_features,patch_features[0]) #patch-level，最后一层
                attn_features = torch.mean(attn_features,dim=1)
                # ## reference作为Key
                label = items["anomaly"] 
                image_features = image_features/image_features.norm(dim=-1, keepdim=True)  ###[batch_size,dim]
                text_features = text_features/text_features.norm(dim=-1, keepdim=True)  ###[batch_size,2,dim]
                attn_features = attn_features/attn_features.norm(dim=-1, keepdim=True) ###[batch_size,dim]
                print(f"image_features dtype: {image_features.dtype}")
                print(f"text_features dtype: {text_features.dtype}")
                text_probs = image_features @ text_features/0.07
                print("text_probs shape:", text_probs.shape)  # 在squeeze之前
                print("squeezed text_probs shape:", text_probs.squeeze().shape)
                print("label shape:", label.shape)

                #print("text_probs: {}".format(text_probs.size())) #[8,2] normal的index为0，abnormal index为1
                refer_probs =  prompt_learner.classfier(attn_features)

                anomaly_maps = []
                #for idx in range(len(patch_features)):
                    #print(patch_features[idx].size()) # [8, 1369, 1024]
                normalized_patch = patch_features[0] / patch_features[0].norm(dim=-1, keepdim=True)
                patch_feature = prompt_learner.image_proj(normalized_patch) 
                anomaly_map = (100.0 * patch_feature @ text_features)
                B, L, C = anomaly_map.shape
                H = int(np.sqrt(L))
                anomaly_map = F.interpolate(anomaly_map.permute(0, 2, 1).view(B, 2, H, H),
                                            size=image_size, mode='bilinear', align_corners=True)
                anomaly_map = torch.softmax(anomaly_map, dim=1)
                anomaly_maps.append(anomaly_map)

            image_loss = F.cross_entropy(text_probs, label.to(device))
            reference_loss = F.cross_entropy(refer_probs, label.to(device))
            #loss = (image_loss + reference_loss)/2.
            
            print("image_loss:{} ; reference_loss: {}".format(image_loss,reference_loss))
            gt = items['img_mask'].squeeze().to(device)
            gt[gt > 0.5], gt[gt <= 0.5] = 1, 0
            loss = 0.
            for i in range(len(anomaly_maps)):
                loss += loss_focal(anomaly_maps[i], gt)
                loss += loss_dice(anomaly_maps[i][:, 1, :, :], gt) 
                loss += loss_dice(anomaly_maps[i][:, 0, :, :], 1-gt)

            optimizer.zero_grad()
            (loss+image_loss+reference_loss).backward()
            optimizer.step()
            loss_list.append(loss.item())
            # if idxs % 100 == 0:
            #     wandb.log({"map_loss": loss.item(), "image_loss": image_loss.item(),"reference_loss": reference_loss.item()}, step=epoch * len(train_dataloader) + idxs)
            if idxs % 100 == 0:
                writer.add_scalar('map_loss', loss, epoch * len(train_dataloader) + idxs)  # 记录损失
                writer.add_scalar('image_loss', image_loss, epoch * len(train_dataloader) + idxs)  # 记录损失
                writer.add_scalar('reference_loss', reference_loss, epoch * len(train_dataloader) + idxs)  # 记录损失
                writer.flush()
        # logs
        if (epoch + 1) % args.print_freq == 0:
            logger.info('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, np.mean(loss_list)))

        # save model
        if (epoch + 1) % args.save_freq == 0:
            ckp_path = os.path.join(save_path, 'epoch_' + str(epoch + 1) + '.pth')
            torch.save({"prompt_learner": prompt_learner.state_dict()}, ckp_path)

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
    parser.add_argument("--features_list", type=int, nargs="+", default=[24], help="features used")
    # hyper-parameter
    parser.add_argument("--epoch", type=int, default=200, help="epochs")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="batch size")
    parser.add_argument("--image_size", type=int, default=224, help="image size")
    parser.add_argument("--aug_rate", type=float, default=0.2, help="image size")
    parser.add_argument("--print_freq", type=int, default=30, help="print frequency")
    parser.add_argument("--save_freq", type=int, default=3, help="save frequency")
    parser.add_argument("--k_shot", type=int, default=10, help="e.g., 10-shot, 5-shot, 1-shot")
    parser.add_argument("--n_ctx", type=int, default=12, help="e.g., 5,10,20")

    args = parser.parse_args()

    setup_seed(111)
    train(args)

