import os
from typing import Union, List
from pkg_resources import packaging
import torch
import numpy as np
from open_clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
# from open_clip import tokenizer
# simple_tokenizer = tokenizer.SimpleTokenizer()
from copy import deepcopy
import torch.nn as nn

from collections import OrderedDict

_tokenizer = _Tokenizer()


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[
    torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def _get_clones(module, N):
    return nn.ModuleList([deepcopy(module) for i in range(N)])


# "candle",
# "capsules",
# "cashew",
# "chewinggum",
# "fryum",
# "macaroni1",
# "macaroni2",
# "pcb1",
# "pcb2",
# "pcb3",
# "pcb4",
# "pipe_fryum",

class FewVand_PromptLearner(nn.Module):
    def __init__(self, clip_model, n_ctx, device,Pparameters):
        super().__init__()
        classnames = ["obj"]
        self.n_cls = len(classnames) ## 所有类别均采用相同的text prompt
        self.n_ctx = n_ctx
        n_ctx_pos = self.n_ctx
        n_ctx_neg = self.n_ctx
        # self.text_encoder_n_ctx = design_details["learnabel_text_embedding_length"]
        ctx_init_pos = ""
        ctx_init_neg = ""
        dtype = clip_model.transformer.get_cast_dtype()

        ctx_dim = clip_model.ln_final.weight.shape[0]

        self.classnames = classnames

        self.state_normal_list = [
            "{}",
        ]

        self.state_anomaly_list = [
            "{}",
        ]

        normal_num = len(self.state_normal_list)
        anormaly_num = len(self.state_anomaly_list)
        self.normal_num = normal_num
        self.anormaly_num = anormaly_num
        self.num_query_tokens = Pparameters['num_query_tokens']
        self.width = Pparameters["width"]
        self.emb_dim = Pparameters["emb_dim"]
        self.query_token =  nn.Parameter(
    torch.randn(self.num_query_tokens, self.width).to(device)
    ) 
        self.image_proj = nn.Linear(self.width,self.emb_dim).to(device)
        self.classfier = nn.Linear(self.width,2).to(device)

        ctx_vectors_pos = torch.empty(n_ctx_pos, ctx_dim, dtype=dtype)
        ctx_vectors_neg = torch.empty(n_ctx_neg, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors_pos, std=0.02)
        nn.init.normal_(ctx_vectors_neg, std=0.02)
        prompts_pos = " ".join(["A"] * n_ctx_pos)+"."
        prompts_neg = " ".join(["N"] * n_ctx_neg)+'.'

        self.ctx_pos = nn.Parameter(ctx_vectors_pos).to(device)  # to be optimized
        self.ctx_neg = nn.Parameter(ctx_vectors_neg).to(device)  # to be optimized

        tokenized_prompts_pos = tokenize(prompts_pos).to(device)
        tokenized_prompts_neg = tokenize(prompts_neg).to(device)

        # 生成相应的text embedding
        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(tokenized_prompts_pos).type(dtype)
            embedding_neg = clip_model.token_embedding(tokenized_prompts_neg).type(dtype)
            print("embedding_pos", embedding_pos.shape) #torch.Size([1, 77, 768])

        self.register_buffer("token_prefix_pos", embedding_pos[:,:1, :])
        self.register_buffer("token_suffix_pos", embedding_pos[:,1 + n_ctx_pos:, :])
        self.register_buffer("token_prefix_neg", embedding_neg[:,:1, :])
        self.register_buffer("token_suffix_neg", embedding_neg[:,1 + n_ctx_neg:, :])
  
        self.register_buffer("tokenized_prompts_pos", tokenized_prompts_pos)
        self.register_buffer("tokenized_prompts_neg", tokenized_prompts_neg)
        print("tokenized_prompts shape", self.tokenized_prompts_pos.shape, self.tokenized_prompts_neg.shape)
        #[1,77]
    def forward(self, cls_id=None):
        ctx_pos = self.ctx_pos
        ctx_neg = self.ctx_neg
        # print("shape", self.ctx_pos[0:1].shape, ctx_pos.shape)
        prefix_pos = self.token_prefix_pos
        prefix_neg = self.token_prefix_neg
        suffix_pos = self.token_suffix_pos
        suffix_neg = self.token_suffix_neg

        print("prefix_pos:",prefix_pos.size(),ctx_pos.size(),suffix_pos.size())
#prefix_pos: torch.Size([1, 1, 768])
#  torch.Size([12, 768]) 
# torch.Size([1, 64, 768])

        prompts_pos = torch.cat(
            [
                # N(the number of template), 1, dim
                prefix_pos.squeeze(0),  # ( 1, dim)
                ctx_pos,  # (n_ctx, dim)
                suffix_pos.squeeze(0),  # (n_cls, *, dim)
            ],
            dim=0,
        )

        prompts_neg = torch.cat(
            [
                prefix_neg.squeeze(0),  
                ctx_neg, 
                suffix_neg.squeeze(0),  
            ],
            dim=0,
        )
        prompts = torch.stack((prompts_neg, prompts_pos), dim=0) #[2,77,768]
        print("prompts:",prompts.size())

        tokenized_prompts = torch.stack((self.tokenized_prompts_neg, self.tokenized_prompts_pos), dim=0) #[2,77]

        print("tokenized_prompts:",tokenized_prompts.size())

        prompt = prompts
        tokenized_prompt = tokenized_prompts.squeeze(1)

        
        return prompt, tokenized_prompt

    
def encode_text_with_prompt_ensemble2(model, prompt_learner,objs=["objs"]):
    text_prompts = {}
    for idx, obj in enumerate(objs):
        prompt, tokenized_prompt = prompt_learner(idx)
        print("prompt",prompt.size(),tokenized_prompt.size())
        # [2, 77, 768], [2, 77]
        text_features = model.encode_text_learn(prompt, tokenized_prompt) # [2,768]
        text_features = text_features.permute(1, 0)
        text_prompts[obj] = text_features
    if objs == ["objs"]:
        return text_prompts["objs"]
    else:
        return text_prompts