from torch import Tensor, nn
import torch
from torch.nn import functional as F
### 第一步修改prompt， 正常-异常prompt进行拼接， 初始化N个learnable 的toknens（wc）
## 第二步只使用cls的表征 (wc)
## 第三步引入注意力机制 
# (1) 如何引入k-shot（重点）。（2）在什么地方计算attention （3）attention之后通过一个二分类器，然后loss应该如何修改 
# （4）训练的时候如何指定哪些参数可以更新
## loss还没有解决怎么写！！！
class LinearLayer(nn.Module):
    def __init__(self, dim_in, dim_out, k, model):
        # k 代表feature list的长度 features_list 6 12 18 24
        super(LinearLayer, self).__init__()
        if 'ViT' in model:
            self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for i in range(k)])
        else:
            ## 否则输入维度就会变
            self.fc = nn.ModuleList([nn.Linear(dim_in * 2 ** (i + 2), dim_out) for i in range(k)])

    def forward(self, tokens):
        for i in range(len(tokens)):
            if len(tokens[i].shape) == 3: ##[b,len_patch,dim] [8,1370,1024]
                tokens[i] = self.fc[i](tokens[i][:, 1:, :])  # ##[b,len_patch,dim] [8,1369,768]
            else:
                B, C, H, W = tokens[i].shape
                tokens[i] = self.fc[i](tokens[i].view(B, C, -1).permute(0, 2, 1).contiguous())
        return tokens

class Classifier(nn.Module):
    def __init__(self, dim_in, dim_out):
        # k 代表feature list的长度 features_list 6 12 18 24
        super(Classifier, self).__init__()
        
           
        self.fc =nn.Linear(dim_in,dim_out)

    def forward(self, x):
        x = x.to(dtype=torch.float32)
        output = self.fc(x)
        return output


class linearlayer(nn.Module):
    def __init__(self, dim_in, dim_out):
        # 对text prompt 进行线性变换 --768-->1024
        super(linearlayer, self).__init__()
        
        self.fc =nn.Linear(dim_in,dim_out)

    def forward(self, x):
        x = x.to(dtype=torch.float32)
        output = self.fc(x) ##
        #output = torch.sigmoid(output) ## 进行sigmoid计算
        return output


class linearlayer_att(nn.Module):
    def __init__(self, dim_in, dim_out):
        # 对attention feature进行线性变换 --1024-->1
        super(linearlayer_att, self).__init__()
        
        self.fc =nn.Linear(dim_in,dim_out)

    def forward(self, x):
        x = x.to(dtype=torch.float32)
        output = self.fc(x) ##
        output = torch.sigmoid(output) ## 进行sigmoid计算
        return output