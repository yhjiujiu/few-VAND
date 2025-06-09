import torch
import torch.nn as nn
import open_clip
from typing import Tuple

class DualTokenTextEncoder(nn.Module):
    def __init__(
        self,
        device,
        clip_model: nn.Module,
        num_normal_tokens: int = 5,
        num_abnormal_tokens: int = 5,
        freeze_base: bool = True,
        
    ):
        """
        双分支可学习token编码器
        
        参数:
            clip_model: 预加载的OpenCLIP模型
            num_normal_tokens: 正常样本token数量
            num_abnormal_tokens: 非正常样本token数量
            freeze_base: 冻结基础模型参数
        """
        super().__init__()
        self.text = clip_model
        #self.text = clip_model.encode_text
        
        # 共享基础组件
        self.token_embedding = self.text.token_embedding
        self.positional_embedding = self.text.positional_embedding
        self.transformer = self.text.transformer
        self.ln_final = self.text.ln_final
        self.text_projection = self.text.text_projection
        self.attn_mask = self.text.attn_mask
        self.device = device

        #print("self.attn_mask: {}".format(self.attn_mask))
        
        # 冻结基础模型
        if freeze_base:
            for param in self.text.parameters():
                param.requires_grad = False
        
        # 创建两组可学习token
        embed_dim = self.token_embedding.weight.shape[1]
        
        # 正常样本token (正态分布初始化)
        self.normal_tokens = nn.Parameter(
            torch.randn(1, num_normal_tokens, embed_dim) * 0.02
        )
        
        # 非正常样本token (均匀分布初始化，增强差异性)
        self.abnormal_tokens = nn.Parameter(
            torch.empty(1, num_abnormal_tokens, embed_dim)
        )
        nn.init.uniform_(self.abnormal_tokens, -0.05, 0.05)
        #print("self.abnormal_tokens: {}".format(self.abnormal_tokens))
        
        # 位置编码适配器
        self.normal_pos_emb = self._get_positional_embedding(num_normal_tokens)
        self.abnormal_pos_emb = self._get_positional_embedding(num_abnormal_tokens)
        
    
    def _get_positional_embedding(self, num_tokens: int) -> torch.Tensor:
        """获取适配的位置编码"""
        return self.positional_embedding[:num_tokens].unsqueeze(0)
    
    def _encode_tokens(self, tokens: torch.Tensor, pos_emb: torch.Tensor) -> torch.Tensor:
        """处理单个token组的编码流程"""
        # 嵌入 + 位置编码
        #print("self.attn_mask: {}".format(self.attn_mask.size())) [5,5]
        x = tokens + pos_emb
        x = x.to(self.device)
        #print("x: {}".format(x.size())) ##[1,5,768]
        attn_mask = build_causal_attn_mask(x.size(1)).to(self.device)
        # Transformer处理
        x = x.permute(1, 0, 2)  # [N, B, D] 序列长度，批次大小，维度
        x, att, out_list = self.transformer(x, attn_mask=attn_mask)
        
        x = x.permute(1, 0, 2)  # [B, N, D]
        #print("x: {}".format(x.size())) ##[1,5,768] 
        # 层归一化
        x = self.ln_final(x)
        
        # 特征池化 (取CLS token)
        return x[:, 0, :]  # [B, D]
    
    def forward(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播 - 处理双分支token并生成拼接特征
        
        返回:
            normal_feature: 正常样本特征 [1, D]
            abnormal_feature: 非正常样本特征 [1, D]
            fused_feature: 拼接特征 [2, D]
        """
        # 1. 分别编码两个分支
        normal_feature = self._encode_tokens(self.normal_tokens, self.normal_pos_emb)
        abnormal_feature = self._encode_tokens(self.abnormal_tokens, self.abnormal_pos_emb)
        
        # 2. 应用文本投影
        normal_feature = normal_feature @ self.text_projection
        abnormal_feature = abnormal_feature @ self.text_projection
        
        # 3. 特征拼接
        # [1, output_dim]
        #print("normal_feature: {}".format(normal_feature.size()))  [1, 768]
        #print("abnormal_feature: {}".format(abnormal_feature.size())) [1, 768]

        concatenated_feature = torch.stack([normal_feature, abnormal_feature], dim=1)
        #print("concatenated_feature: {}".format(concatenated_feature.size())) #[1, 2, 768]
         
         ## # [1,2, output_dim] 因为每次的batch_size 等于1，在训练的时候可以通过复制B次，得到batch的数据
        return concatenated_feature

# ==================== 使用示例 ====================
# 1. 加载基础模型
# model, _, _ = open_clip.create_model_and_transforms("ViT-B-32")

# # 2. 创建双分支编码器
# dual_encoder = DualTokenTextEncoder(
#      clip_model=model,
#      num_normal_tokens=6,      # 6个正常token
#      num_abnormal_tokens=4,    # 4个异常token
#      freeze_base=True
# )



def build_causal_attn_mask(seq_len: int) -> torch.Tensor:
    """
    构造一个因果 attention mask，左下为0，右上为 -inf。
    shape: [seq_len, seq_len]
    
    返回:
        一个 float 型 Tensor，适用于多头注意力
    """
    mask = torch.full((seq_len, seq_len), float('-inf'))
    mask = torch.triu(mask, diagonal=1)  # 上三角为 -inf，包含右上角
    mask = mask.masked_fill(mask == float('-inf'), float('-inf'))  # 保证类型为float
    mask = mask.masked_fill(mask == 0, 0.0)
    return mask
