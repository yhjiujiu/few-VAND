import torch
from dataset import VisaDataset, MVTecDataset
import torch
import torch.nn.functional as F

def memory(model_name, model, obj_list, dataset_dir, save_path, preprocess, 
transform, k_shot,dataset_name, device):
    mem_features = {}
    ## obj_list 为测试集的cls name.
    for obj in obj_list:
        if dataset_name == 'mvtec':
            data = MVTecDataset(root=dataset_dir, transform=preprocess, target_transform=transform,
                                aug_rate=-1, mode='train', k_shot=k_shot, save_dir=save_path, obj_name=obj)
        else:
            data = VisaDataset(root=dataset_dir, transform=preprocess, target_transform=transform,
                               mode='train', k_shot=k_shot, save_dir=save_path, obj_name=obj)
        dataloader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
        features = []
        for items in dataloader:
            image = items['img'].to(device)
            with torch.no_grad():
                image_features, patch_token_ = model.encode_image(image, [24])
                if 'ViT' in model_name:
                    patch_tokens = patch_token_[0][0, 1:, :] ##取除cls的其他所有token得编码
                else: 
                    pass
                ##  patch_token_： [batch_size, num_patches, hidden_dim] 
                features.append(image_features) # [batch,dim]

        img_features = torch.stack(features, dim=0).squeeze(1) #[4,dim]
        mem_features[obj]= img_features ## 每个对象下的reference image 编码

    return mem_features


def memory_pixel(model_name, model, obj_list, dataset_dir, save_path, preprocess, 
transform, k_shot,dataset_name, device):
    mem_features = {}
    ## obj_list 为测试集的cls name.
    for obj in obj_list:
        if dataset_name == 'mvtec':
            data = MVTecDataset(root=dataset_dir, transform=preprocess, target_transform=transform,
                                aug_rate=-1, mode='train', k_shot=k_shot, save_dir=save_path, obj_name=obj)
        else:
            data = VisaDataset(root=dataset_dir, transform=preprocess, target_transform=transform,
                               mode='train', k_shot=k_shot, save_dir=save_path, obj_name=obj)
        dataloader = torch.utils.data.DataLoader(data, batch_size=1, shuffle=False)
        features = []
        for items in dataloader:
            image = items['img'].to(device)
            with torch.no_grad():
                _, patch_token_ = model.encode_image(image, [24])
                if 'ViT' in model_name:
                    patch_tokens = torch.cat([p[0, 1:, :] for p in patch_token_], dim=0) # ##取除cls的其他所有token得编码
                else:
                    pass
                ##  patch_token_： [batch_size, num_patches, hidden_dim]  batch_size=1
                features.append(patch_tokens) # [batch,num_patches,dim]

        patch_features = torch.stack(features, dim=0).squeeze(1) #[4,num_patches,dim]
        patch_features = torch.mean(patch_features,dim=0)  # [num_patches,dim]
        mem_features[obj]= patch_features ## 每个对象下的reference image 编码

    return mem_features

def attention(Q, K, V):
    """
    计算注意力输出
    :param Q: 查询矩阵，维度为 (1, d1)
    :param K: 键矩阵，维度为 (batch_size, d1)
    :param V: 值矩阵，维度为 (batch_size, d1)
    :return: 注意力输出，维度为 (batch_size, d1)
    """
    batch_size = K.size(0)
    
    # 重复 Q，维度为 (batch_size, d1)
    Q_repeated = Q.repeat(batch_size, 1)
    
    # 计算注意力权重
    attention_weights = F.softmax(torch.matmul(Q_repeated, K.T) / (K.size(1) ** 0.5), dim=-1)
    
    # 计算注意力输出
    attention_output = torch.matmul(attention_weights, V)
    
    return attention_output


def patch_attention(Q, K, V):
    """
    计算注意力输出
    :param Q: 查询矩阵，维度为 (num_query, d1)
    :param K: 键矩阵，维度为 (batch_size, num_query,d1)
    :param V: 值矩阵，维度为 (batch_size, num_query,d1)
    :return: 注意力输出，维度为 (batch_size, num_query,d1)
    """
    # 确保输入维度匹配
    assert Q.size(1) == K.size(2) == V.size(2), "输入特征维度d1必须一致"
    
    # 计算缩放因子（特征维度的平方根）
    d_k = Q.size(1)
    scale_factor = torch.sqrt(torch.tensor(d_k, dtype=torch.float))
    
    # 扩展Q的维度以匹配K的batch_size [num_query, d1] -> [batch_size, num_query, d1]
    Q_expanded = Q.unsqueeze(0).expand(K.size(0), -1, -1)
    
    # 计算Q和K的点积注意力分数 [batch_size, num_query, num_query]
    attn_scores = torch.bmm(Q_expanded, K.transpose(1, 2)) / scale_factor
    
    # 应用softmax获取注意力权重
    attn_weights = F.softmax(attn_scores, dim=-1)
    
    # 使用注意力权重加权值矩阵V
    output = torch.bmm(attn_weights, V)
    
    
    return output
