import torch
from dataset import VisaDataset, MVTecDataset
import torch
import torch.nn.functional as F

def memory(model_name, model, obj_list, dataset_dir, save_path, preprocess, 
transform, k_shot,dataset_name, device):
    mem_features = {}
    ## obj_list 为测试集的cls name.
    #print("obj_list: {}".format(obj_list[:3]))
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
                #print("patch_tokens: {}".format(patch_tokens.size()))
                features.append(patch_tokens) # [len,dim]

                #image_features = model.encode_image(image) ## 图片整体特征
            #features.append(image_features)
        img_features = torch.stack(features, dim=0) #[4,len,dim]
        #print("img_features: {}".format(img_features.size())) #img_features: torch.Size([4,  768])
        mem_features[obj]= img_features ## 每个对象下的reference image 编码

    return mem_features



def attention_single_query(Q, K, V):
    """
    单查询注意力计算：Q 是单一查询（如 decoder 查询一个位置），
    对每个 batch 的 K/V 进行 attention。

    参数:
        Q: [len, d1]，如 [1039, d1]，单个查询序列（reference）
        K: [batch_size, seq_len, d1]
        V: [batch_size, seq_len, d1]
Q: torch.Size([1, 1369, 1024]), K: torch.Size([8, 1369, 1024]), V: torch.Size([1369, 1024])

    返回:
        context: [batch_size, seq_len, d1]，与 V 相同形状
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
