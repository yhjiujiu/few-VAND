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
                features.append(image_features) # [batch,dim]

        img_features = torch.stack(features, dim=0).squeeze(1) #[4,dim]
        mem_features[obj]= img_features ## 每个对象下的reference image 编码

    return mem_features



def attention_single_query(Q, K, V):
    """
    单查询注意力计算：Q 是单一查询（如 decoder 查询一个位置），
    对每个 batch 的 K/V 进行 attention。

    参数:
        Q: [1, d1], 
        K: [batch_size, d1]
        V: [batch_size, d1]
    """
    # Q: [seq_len, d1] -> [1, seq_len, d1]（为了 broadcast）
    Q = Q.unsqueeze(0)  # [1, seq_len, d1]

    #print("Q: {}, K: {}, V: {}".format(Q.size(),K.size(),V.size()))
    # Attention logits: [batch_size, seq_len, seq_len]
    attn_logits = torch.matmul(K, Q.transpose(-1, -2))  # [batch_size, seq_len, seq_len]

    # Scale by sqrt(d1)
    d1 = Q.size(-1)
    attn_logits = attn_logits / (d1 ** 0.5)

    # Attention weights: softmax over last dim
    attn_weights = F.softmax(attn_logits, dim=-1)  # [batch_size, seq_len, seq_len]

    # Apply attention to V
    context = torch.matmul(attn_weights, V)  # [batch_size, seq_len, d1]

    return context

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