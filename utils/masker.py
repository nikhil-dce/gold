import os
import random
import time
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pdb

from utils.help import prepare_inputs
from torch.utils.data import TensorDataset

from utils.evaluate import compute_centroids, make_covariance_matrix, process_diff
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Keyword():
    def __init__(self, keyword_type, keyword):
        self.keyword_type = keyword_type
        self.keyword = keyword

    def __len__(self):
        return len(self.keyword)

def backbone(args, output_attentions = False):
    if args.model == 'bert':
        from transformers import BertModel, BertTokenizer
        backbone = BertModel.from_pretrained('bert-base-uncased', output_attentions=output_attentions)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        tokenizer.name = 'bert-base-uncased'
        special = { 'additional_special_tokens': ['<customer>', '<agent>', '<kb>']  }
        tokenizer.add_special_tokens(special)
        backbone.resize_token_embeddings(len(tokenizer))
    if args.model == 'roberta':
        from transformers import RobertaModel, RobertaTokenizer
        backbone = RobertaModel.from_pretrained('roberta-base', output_attentions=output_attentions)
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base', add_prefix_space=True)
        tokenizer.name = 'roberta-base'
        special = { 'additional_special_tokens': ['<customer>', '<agent>', '<kb>']  }
        tokenizer.add_special_tokens(special)
        backbone.resize_token_embeddings(len(tokenizer))
    return backbone, tokenizer

def get_keywords(args, dataset, mahala = 0):

    attn_model, tokenizer = backbone(args, True)
    attn_model.to(device)  # only backbone
    CKPT_PATH = "/work/ds448/gold_mod/gold/results/star/baseline"
    # args.best_base_model "/work/ds448/gold_mod/gold/results/star/baseline/epoch12_star_lr1e-05_acc688.pt"
    attn_model_path = args.best_base_model
    #assert args.attn_model_path is not None
    state_dict = torch.load(os.path.join(CKPT_PATH, "star", attn_model_path))

    new_state_dict = dict()
    for key, value in state_dict.items():  # only keep backbone parameters
        print(key)
        if key.split('.')[0] == 'encoder':
            key = '.'.join(key.split('.')[1:])  # remove 'backbone'
            new_state_dict[key] = value
    
    attn_model.load_state_dict(new_state_dict)  # backbone state dict

    #if torch.cuda.device_count() > 1:
    #    attn_model = nn.DataParallel(attn_model)
    if mahala == 1:
        keyword = get_attention_keyword_mahalanobis(args, dataset, tokenizer, attn_model)
        keyword_path = "/work/ds448/gold_mod/gold/utils/keyword_" + args.model + '_' + "mahala_" + args.task + "_10perclass.pth"
    elif mahala == 0:
        keyword = get_attention_keyword(args, dataset, tokenizer, attn_model)
        keyword_path = "/work/ds448/gold_mod/gold/utils/keyword_" + args.model + '_'  + args.task + "_10perclass.pth"
    if mahala == 2:
        keyword = get_attention_keyword_mahalanobis(args, dataset, tokenizer, attn_model)
        keyword_path = "/work/ds448/gold_mod/gold/utils/keyword_" + args.model + '_' + "mahalaV2_" + args.task + "_10perclass.pth"
    keyword = Keyword('attention', keyword)
    
    torch.save(keyword, keyword_path)
    print("Keywords saved in" + keyword_path)
    #
    #_, whole_batch = fullloader[0]
    #inputs, labels = prepare_inputs(whole_batch, attn_model)
    masked_dataset(args, tokenizer, dataset, keyword.keyword, mahala, seed = 0.1, key_mask_ratio= 0.5) ##seed 

def get_attention_keyword_mahalanobis(args, dataset, tokenizer, attn_model, keyword_per_class=10):
    #loader = DataLoader(dataset.train_dataset, shuffle=False,
     #                   batch_size=16, num_workers=4)
    loader = dataset
    
    SPECIAL_TOKENS = tokenizer.all_special_ids
    PAD_TOKEN = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    vocab_size = len(tokenizer)

    attn_score = torch.zeros(vocab_size)
    attn_freq = torch.zeros(vocab_size)

    data_stack = []
    labels_stack = []
    for _, batch in enumerate(loader):
        inputs, labels = prepare_inputs(batch, attn_model)
        with torch.no_grad():
            sequence = attn_model(**inputs)[0]
            hidden = sequence[:, 0, :]
            data_stack.append(hidden.detach().cpu())
            labels_stack.append(labels.detach().cpu())
    data_stack = torch.cat(data_stack)
    labels_stack = torch.cat(labels_stack)
    centroids = compute_centroids(data_stack, labels_stack)
    inv_cov_matrix = make_covariance_matrix(args, data_stack, centroids, labels_stack)

    print("Stacking done!")

    max_mahala = 0 ; min_mahala = 1e7
    mahala_map = {}

    for batch_index, batch in enumerate(loader):
        inputs, labels = prepare_inputs(batch, attn_model)
        with torch.no_grad():
            #out_h, out_p, attention_layers = attn_model(tokens_a, tokens_b, tokens_c)
            #attention_layers = attn_model(tokens_a, tokens_b, tokens_c, tokens_d)
            attention_layers = attn_model(**inputs)
        sequence = attention_layers[0]
        hidden = sequence[:, 0, :]
        attention = attention_layers[-1][0]  # attention of final layer (batch_size, num_heads, max_len, max_len)
        mahala_distance = process_diff(args, centroids, inv_cov_matrix, hidden.cpu(), None, None)[0]
        batch_size = attention.size(0)
        for i in range(attention.size(0)):  # batch_size
            mahala_map[i + batch_index*batch_size] = mahala_distance[i].item()
            div = mahala_distance[i].item()
            if div > max_mahala:
                max_mahala = div
            if div < min_mahala:
                min_mahala = div


    for batch_index , batch in enumerate(loader):
        
        #tokens_a = tokens[0].to(device)
        #tokens_b = tokens[1].to(device)
        #tokens_c = tokens[2].to(device)
        #tokens_d = tokens[3].reshape((16,1)).to(device)
        inputs, labels = prepare_inputs(batch, attn_model)
        #pdb.set_trace()
        #tokens = tokens.to(device)
        with torch.no_grad():
            #out_h, out_p, attention_layers = attn_model(tokens_a, tokens_b, tokens_c)
            #attention_layers = attn_model(tokens_a, tokens_b, tokens_c, tokens_d)
            attention_layers = attn_model(**inputs)
        sequence = attention_layers[0]
        hidden = sequence[:, 0, :]
        attention = attention_layers[-1][0]  # attention of final layer (batch_size, num_heads, max_len, max_len)
        # mahala_distance = process_diff(args, centroids, inv_cov_matrix, hidden.cpu(), None, None)[0]
        # pdb.set_trace()
        # for batch_index in range(len(mahala_distance)):
        #     torch.autograd.grad(mahala_distance[batch_index], attention[batch_index])
        # attention
        # 1. Get everything -  seq, pooled, attn representations. torch variable of attention, get hidden through sequence, stack for entire data, N x D, stack labels N x n
        # 2. Make_clusters vectors N x D, labels N x n
        # 3. First loop and stack. Second loop - get hidden of batch and calculate mahalanobis distance, use process_diff to calculate batch distance with entire cluster
        # 4. In second loop, attention scores, also mahala dist, and set gradient true for attention score. Set attention, set_grad = True, do backward
        attention = attention.sum(dim=1)  # sum over attention heads (batch_size, max_len, max_len)
        # pdb.set_trace()
        batch_size = attention.size(0)
        # for i in range(attention.size(0)):  # batch_size
        #     mahala_map[i + batch_index*batch_size] = mahala_distance[i].item()
        #     if div > max_div:
        #         max_div = div
        #     if div < min_div:
        #         min_div = div
                
        for i in range(attention.size(0)):  # batch_size
            for j in range(attention.size(-1)):  # max_len
                #pdb.set_trace()
                token = inputs['input_ids'][i][j].item()

                if token == PAD_TOKEN: # token == pad_token
                    break
                
                if token in SPECIAL_TOKENS:  # skip special token
                    continue

                score = attention[i][0][j]  # 1st token = CLS token

                # if mahala_distance[i].item() == 0.0:
                #     div = torch.mean(mahala_distance).item() 
                #     # pdb.set_trace()
                # else:
                #     div = mahala_distance[i].item()
                
                norm_mahala = (mahala_map[i + batch_size * batch_index] - min_mahala)/(max_mahala - min_mahala)
                attn_score[token] += score.item()/norm_mahala #to-do max and min div
                attn_freq[token] += 1
        # break
    
    for tok in range(vocab_size):
        if attn_freq[tok] == 0:
            attn_score[tok] = 0
        else:
            attn_score[tok] /= attn_freq[tok]  # normalize by frequency
    n_classes = 150
    if args.task == 'flow':
        n_classes = 48
    if args.task == 'rostd':
        n_classes = 12
    num = keyword_per_class * n_classes  # 150, 48, 12 number of total keywords hardcoded have to be replaced
    keyword = attn_score.argsort(descending=True)[:num].tolist()
    return keyword


def get_attention_keyword(args, dataset, tokenizer, attn_model, keyword_per_class=10):
    #loader = DataLoader(dataset.train_dataset, shuffle=False,
     #                   batch_size=16, num_workers=4)
    loader = dataset
    SPECIAL_TOKENS = tokenizer.all_special_ids
    PAD_TOKEN = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)

    vocab_size = len(tokenizer)

    attn_score = torch.zeros(vocab_size)
    attn_freq = torch.zeros(vocab_size)

    for _, batch in enumerate(loader):
        #pdb.set_trace()
        #tokens_a = tokens[0].to(device)
        #tokens_b = tokens[1].to(device)
        #tokens_c = tokens[2].to(device)
        #tokens_d = tokens[3].reshape((16,1)).to(device)
        inputs, labels = prepare_inputs(batch, attn_model)
        #pdb.set_trace()
        #tokens = tokens.to(device)
        with torch.no_grad():
            #out_h, out_p, attention_layers = attn_model(tokens_a, tokens_b, tokens_c)
            #attention_layers = attn_model(tokens_a, tokens_b, tokens_c, tokens_d)
            attention_layers = attn_model(**inputs)
        attention = attention_layers[-1][0]  # attention of final layer (batch_size, num_heads, max_len, max_len)
        # 1. Get everything -  seq, pooled, attn representations. torch variable of attention, get hidden through sequence, stack for entire data, N x D, stack labels N x n
        # 2. Make_clusters vectors N x D, labels N x n
        # 3. First loop and stack. Second loop - get hidden of batch and calculate mahalanobis distance, use process_diff to calculate batch distance with entire cluster
        # 4. In second loop, attention scores, also mahala dist, and set gradient true for attention score. Set attention, set_grad = True, do backward
        attention = attention.sum(dim=1)  # sum over attention heads (batch_size, max_len, max_len)
        #pdb.set_trace()
        for i in range(attention.size(0)):  # batch_size
            for j in range(attention.size(-1)):  # max_len
                #pdb.set_trace()
                token = inputs['input_ids'][i][j].item()

                if token == PAD_TOKEN: # token == pad_token
                    break
                
                if token in SPECIAL_TOKENS:  # skip special token
                    continue

                score = attention[i][0][j]  # 1st token = CLS token

                attn_score[token] += score.item()
                attn_freq[token] += 1
        # break
    for tok in range(vocab_size):
        if attn_freq[tok] == 0:
            attn_score[tok] = 0
        else:
            attn_score[tok] /= attn_freq[tok]  # normalize by frequency
    n_classes = 150
    if args.task == 'flow':
        n_classes = 48
    if args.task == 'rostd':
        n_classes = 12
    num = keyword_per_class * n_classes  # 150, 48, 12 number of total keywords hardcoded have to be replaced
    keyword = attn_score.argsort(descending=True)[:num].tolist()

    return keyword

def masked_dataset(args, tokenizer, dataset, keyword, mahala,
                    seed=0, key_mask_ratio=0.5, out_mask_ratio=0.9):
   
    print("Key mask ratio: ", key_mask_ratio)
    #keyword_dict = dict.fromkeys(keyword, i for i in range(len(keyword)))  # convert to dict
    keyword_dict={}
    for i,word in enumerate(keyword):
        keyword_dict[word]=i

    keyword=keyword_dict

    CLS_TOKEN = tokenizer.cls_token_id
    PAD_TOKEN = tokenizer.pad_token_id
    MASK_TOKEN = tokenizer.mask_token_id

    random.seed(seed)  # fix random seed
    
    #tokens = dataset.tensors[0]
    #labels = dataset.tensors[1]

    masked_tokens = []
    masked_labels = []

    attention_masks = []
    input_type_ids = []

    for _, batch in enumerate(dataset):
        inputs, labels = prepare_inputs(batch, _)
        for (token, type_id, mask, label) in zip(inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask'], labels):
            m_token = token.clone()  # masked token (for self-supervision)
            o_token = token.clone()  # outlier token (for entropy regularization)
            m_label = -torch.ones(token.size(0) + 1).long()  # self-sup labels + class label

            for i , tok in enumerate(token):
                if tok == CLS_TOKEN:
                    continue
                elif tok == PAD_TOKEN:
                    break
                if random.random() < key_mask_ratio:  # randomly mask keywords
                    if (keyword is None) or (tok.item() in keyword):  # random MLM or keyword MLM
                        m_token[i] = MASK_TOKEN
                        if keyword is None:
                            m_label[i] = tok  # use full vocabulary
                        else:
                            m_label[i] = keyword[tok.item()] # convert to keyword index

                if (keyword is not None) and (tok.item() not in keyword):
                    if random.random() < out_mask_ratio:  # randomly mask non-keywords
                        o_token[i] = MASK_TOKEN
            # x -> one of the classes , x[i, i+1] -> predicted back ssl, x[j, j+1] -> uniform

            m_label[-1] = label  # class label

            masked_tokens.append(torch.cat([token, m_token, o_token]))  # (original, masked, outlier)
            masked_labels.append(m_label)

            input_type_ids.append(type_id)
            attention_masks.append(mask)

    masked_tokens = torch.stack(masked_tokens)
    masked_labels = torch.stack(masked_labels)

    input_type_ids = torch.stack(input_type_ids)
    attention_masks = torch.stack(attention_masks)

    #pdb.set_trace()
    masked_dataset = TensorDataset(masked_tokens, input_type_ids, attention_masks, masked_labels)
    #masked_tokens Nx768, attention_masks Nx256, 
    if mahala == 1:
        dataset_path = '/work/ds448/gold_mod/gold/utils/masked_dataset_' + args.model + '_'  + "mahala_" + args.task + '_.pt'
    elif mahala == 0:
        dataset_path = '/work/ds448/gold_mod/gold/utils/masked_dataset_' + args.model + '_'  + args.task + '_.pt'
    else:
        dataset_path = '/work/ds448/gold_mod/gold/utils/masked_dataset_' + args.model + '_'  + "mahalaV2_" + args.task + '_.pt'
    
    torch.save(masked_dataset, dataset_path)
    print("Masked dataset saved in " + dataset_path)
    #return masked_dataset, keyword
