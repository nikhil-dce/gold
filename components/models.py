import os, pdb, sys
import numpy as np
import re

import torch
from torch import nn
from torch import optim
from torch.nn import functional as F

from transformers import BertModel, BertConfig, RobertaModel, RobertaConfig
from transformers import AdamW, get_cosine_schedule_with_warmup
from transformers import T5ForConditionalGeneration, BartForConditionalGeneration
from assets.static_vars import baseline_methods, direct_modes, device
# from transformers import GPT2Model

import pdb
from numpy import random
from utils.evaluate import compute_centroids, make_covariance_matrix, process_diff, process_diff_training

class BaseModel(nn.Module):
  # Main model for predicting Unified Meaning Representations of act, topic and modifier
  def __init__(self, args, ontology, tokenizer):
    super().__init__()
    self.version = args.version
    self.ontology = ontology
    self.name = 'basic'

    self.tokenizer = tokenizer
    self.model_setup(args)
    self.verbose = args.verbose
    self.debug = args.debug

    self.classify = nn.Linear(args.embed_dim, 1)
    self.sigmoid = nn.Sigmoid()
    self.criterion = nn.BCEWithLogitsLoss()
 
    self.save_dir = args.save_dir
    self.load_dir = os.path.join(args.output_dir, args.task, 'baseline')
    self.opt_path = os.path.join(self.save_dir, f"optimizer_{args.version}.pt")
    self.schedule_path = os.path.join(self.save_dir, f"scheduler_{args.version}.pt")

  def model_setup(self, args):
    print(f"Setting up {args.model} model")
    if args.method == 'dropout' or args.version == 'augment':
      if args.model == 'bert':
        configuration = BertConfig(hidden_dropout_prob=args.threshold)
        self.encoder = BertModel(configuration)
      elif args.model == 'roberta':
        configuration = RobertaConfig(hidden_dropout_prob=args.threshold)
        self.encoder = RobertaModel(configuration)
    else:
      if args.model == 'bert':
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
      elif args.model == 'roberta':
        self.encoder = RobertaModel.from_pretrained('roberta-base')

    self.encoder.resize_token_embeddings(len(self.tokenizer))  # transformer_check
    self.model_type = args.model

  def forward(self, inputs, targets, outcome='logit'):
    enc_out = self.encoder(**inputs)
    hidden = enc_out['last_hidden_state'][:, 0, :]   # batch_size, embed_dim
    logits = self.classify(hidden).squeeze(1)         # batch_size
    
    loss = self.criterion(logits, targets)
    output = logits if outcome == 'logit' else self.sigmoid(logits)
    return output, loss

  @classmethod
  def from_checkpoint(cls, args, targets, tokenizer, checkpoint_path):
    return cls(args, targets, tokenizer, checkpoint_path)

  def setup_optimizer_scheduler(self, learning_rate, total_steps):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.0 },
        {"params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]

    warmup = int(total_steps * 0.2)
    self.optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, 
                      num_warmup_steps=warmup, num_training_steps=total_steps)

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(self.opt_path) and os.path.isfile(self.schedule_path):
        # Load in optimizer and scheduler states
        self.optimizer.load_state_dict(torch.load(opt_path))
        self.scheduler.load_state_dict(torch.load(schedule_path))

class IntentModel(BaseModel):
  # Main model for predicting the intent of a utterance, rather than binary OOS
  def __init__(self, args, ontology, tokenizer):
    super().__init__(args, ontology, tokenizer)
    if args.task == 'star':
      target_size = len(ontology['regular'])
    elif args.task == 'rostd':
      target_size = len(ontology['finegrain'])
    elif args.task == 'flow':
      allowed = list(ontology.keys())
      allowed.remove('Fence')   # OOS examples are never a valid intent
      target_size = 0
      for category in allowed:
        target_size += len(ontology[category])
    
    #mix-up
    mixup = 0
    if args.mixup == 1:
      mixup = 1
    self.temperature = args.temperature
    self.dropout = nn.Dropout(args.drop_rate)
    self.classify = Classifier(args, target_size)
    self.softmax = nn.LogSoftmax(dim=1)
    self.criterion = nn.CrossEntropyLoss(reduction = 'none')  # combines LogSoftmax() and NLLLoss()
    self.name = 'nlu'
    self.mixedup = mixup

  def forward(self, inputs, targets, outcome='loss'):
    enc_out = self.encoder(**inputs)
    sequence, pooled = enc_out['last_hidden_state'], enc_out['pooler_output']
    return_pre_classifier = outcome in ['nml', 'mahalanobis_preds', 'mahalanobis_nml']
    
    hidden = sequence[:, 0, :] # batch_size, embed_dim

    ###Mix-up
    batch_s = hidden.shape[0]
    if outcome == 'loss' and self.mixedup == 1:
      alpha = 0.2
      index = torch.randperm(batch_s).to(device)
      lambda_v = torch.from_numpy(np.random.beta(
        alpha, alpha, size=(batch_s, 1))).float()
      lambda_v = lambda_v.to(device)

      hidden = lambda_v*hidden + (1.0 - lambda_v) * hidden[index]
      targets_mod = targets[index]   

    if outcome == 'odin':
      noise = torch.randn(hidden.shape) * 1e-6      # standard deviaton of epsilon = 1e-6
      hidden += noise.to(device)
    else:
      hidden = self.dropout(hidden)
    
    pre_classifier = hidden
    if outcome == 'nml':
      pre_classifier, logit = self.classify(hidden, outcome)
    elif outcome == 'mahalanobis_preds':
      pre_classifier, logit = self.classify(hidden, outcome)
    else:
      logit = self.classify(hidden, outcome) # batch_size, num_intents
    
    #pdb.set_trace()
    loss = torch.zeros(batch_s)    # set as default loss\
    loss2 = torch.zeros(batch_s)   # default mixup loss
    if outcome == 'loss':   # used by default for 'intent' and 'direct' training
      
      output = logit  # logit is a FloatTensor, targets should be a LongTensor
      loss = self.criterion(logit, targets) # [batch_size]

      if self.mixedup == 1:
        loss2 = self.criterion(logit, targets_mod)  # [batch_size]
        #pdb.set_trace()

        # Convert lambda_v shape from [batch_size, 1] to [batch_size].
        lambda_v =lambda_v.squeeze() 
        loss = lambda_v * loss + (1 - lambda_v) * loss2

    elif outcome == 'gradient':   # we need to hallucinate a pseudo_label for the loss
      output = logit     # this output will be ignored during the return
      pseudo_label = torch.argmax(logit)
      loss = self.criterion(logit, pseudo_label.unsqueeze(0))
    elif outcome in ['dropout', 'maxprob', 'nml']:
      output = self.softmax(logit)
    elif outcome in ['odin', 'entropy']:
      output = self.softmax(logit / self.temperature)  
    else:                   # used by the 'direct' methods during evaluation
      output = logit
    
    loss = torch.mean(loss)
    if return_pre_classifier:
      return output, loss, pre_classifier
    else:
      return output, loss

class Classifier(nn.Module):
  def __init__(self, args, target_size):
    super().__init__()
    input_dim = args.embed_dim
    self.top = nn.Linear(input_dim, args.hidden_dim)
    self.relu = nn.ReLU()
    self.bottom = nn.Linear(args.hidden_dim, target_size)

  def forward(self, hidden, outcome):
    # hidden has shape (batch_size, 2 * embed_dim)
    middle = self.relu(self.top(hidden))
    # hidden now is (batch_size, hidden_dim)
    logit = self.bottom(middle)
    # logit has shape (batch_size, num_slots)

    if outcome in ['bert_embed', 'mahalanobis', 'gradient', 'mahalanobis_nml']:
      return hidden
      # return middle
    elif outcome == 'mahalanobis_preds':
      return (middle, logit)   
    elif outcome == 'nml':
      norm = torch.linalg.norm(middle, dim=-1, keepdim=True)
      middle_normalized = middle / norm
      logit_normalized = self.bottom(middle_normalized)
      return (middle_normalized, logit_normalized)
      # return (middle, logit)
    else:
      return logit   

    # return middle if outcome in ['bert_embed', 'mahalanobis', 'gradient', 'nml'] else logit

def uniform_labels(labels, n_classes):
    unif = torch.ones(labels.size(0), n_classes).to(device)
    return unif / n_classes

class MaskerIntentModel(IntentModel):
  def __init__(self, args, ontology, tokenizer, vocab_size, centroids, inv_cov_matrix):
      super().__init__(args, ontology, tokenizer)
      self.load_dir = os.path.join(args.output_dir, args.task, 'masker')
      self.net_ssl = nn.Sequential(  # self-supervision layer
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, vocab_size))
      self.dropout = nn.Dropout(0.1)
      self.args = args
      self.dense = nn.Linear(768,768)
      self.centroids = centroids.to(device)
      self.cov_matrix = inv_cov_matrix.to(device)
      #self.binary_classifier= nn.Linear(768, 1)
      
  
  def forward(self, inputs, targets, masked_labels = None, outcome='loss'):
      #labels_ssl = targets[:, :-1]  # self-sup labels (B, K)
      #targets = targets[:, -1]  # class labels (B)
      if outcome != 'loss':
        return super().forward(inputs, targets, outcome)
      else:
        labels_ssl = masked_labels
        
        #masked token same for bert and roberta
        # pdb.set_trace()
        if self.args.model == 'bert':
          enc_out = self.encoder(inputs['input_ids_masked'], inputs['token_type_ids'], inputs['attention_mask'])
        else:
          enc_out = self.encoder(inputs['input_ids_masked'], inputs['attention_mask'])
        sequence, pooled = enc_out['last_hidden_state'], enc_out['pooler_output']
        out_ssl = self.dropout(sequence)
        out_ssl_logits = self.net_ssl(out_ssl)
        out_ssl = out_ssl_logits.permute(0, 2, 1) #16 x vocab x 256
        loss_ssl = F.cross_entropy(out_ssl, labels_ssl, ignore_index=-1)  # ignore non-masks (-1)
        loss_ssl = loss_ssl * 0.000001 #args.lambda_ssl 0.001 default
        # print("loss ssl:", loss_ssl.item())

        #normal
        if self.args.model == 'bert':
            enc_out = self.encoder(inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask'])
            sequence, pooled = enc_out['last_hidden_state'], enc_out['pooler_output']
            hidden = sequence[:, 0, :]
            hidden = self.dropout(hidden)
            batch_s = hidden.shape[0]
            logit = self.classify(hidden, outcome) # batch_size, num_intents
            loss = torch.zeros(batch_s)    # set as default loss\
            output = logit  # logit is a FloatTensor, targets should be a LongTensor
            loss = self.criterion(logit, targets) # [batch_size]
            loss = torch.mean(loss)
            # print("loss:", loss.item())
        if self.args.model == 'roberta':
            enc_out = self.encoder(inputs['input_ids'], inputs['attention_mask'])
            sequence, pooled = enc_out['last_hidden_state'], enc_out['pooler_output']
            # out = self.backbone(x_orig, attention_mask)[0]
            hidden = sequence[:, 0, :] # take cls token (<s>)
            hidden = self.dropout(hidden)
            # hidden = self.dense(hidden)
            # hidden = torch.tanh(hidden)
            # hidden = self.dropout(hidden)
            batch_s = hidden.shape[0]
            logit = self.classify(hidden, outcome) # batch_size, num_intents
            loss = torch.zeros(batch_s)    # set as default loss\
            output = logit  # logit is a FloatTensor, targets should be a LongTensor
            loss = self.criterion(logit, targets) # [batch_size]
            loss = torch.mean(loss)
            print("loss:", loss.item())
      
      
        ### process diff with gpu, return batch_size, loss = max(maha_dist(centroids, hidden, cov_matrix) - epsilon(10^-1), 0) 
        #ood
        if self.args.ood_maha_loss == 1:
          out_ood_logits = None
          if self.args.model == 'bert':
            enc_out = self.encoder(inputs['input_ids_ood'], inputs['token_type_ids'], inputs['attention_mask'])
            sequence, pooled = enc_out['last_hidden_state'], enc_out['pooler_output']  # pooled feature
            # out_ood = self.dropout(pooled)
            hidden_ood = sequence[:, 0, :] #cls to cls comparison for euclidean
            hidden_ood = self.dropout(hidden_ood)
            # hidden_ood = self.dropout(pooled) # pooled instead of cls (masker)
      
            # mahala_distance = process_diff_training(self.args, self.centroids, self.cov_matrix, hidden_ood, None, None)[0]
            # mahala_distance_ind = process_diff_training(self.args, self.centroids, self.cov_matrix, hidden, None, None)[0]

            # margin = torch.ones_like(mahala_distance) * torch.mean(mahala_distance_ind).item() # set manually


            # loss_ent_ood = torch.mean(torch.max(mahala_distance - margin, torch.zeros_like(mahala_distance)))
            # loss_ent_ind = torch.mean(torch.max(mahala_distance_ind - margin, torch.zeros_like(mahala_distance_ind)))
            # loss_ent = -(loss_ent_ood - loss_ent_ind)

            loss_ent =  - torch.mean((hidden_ood - hidden)**2)
            # pdb.set_trace()
            # out_ood_logits = self.classify(hidden, outcome) # batch_size, num_intents
            # out_ood = F.log_softmax(out_ood_logits, dim=1)  # log-probs
            # n_classes = out_ood.shape[1]
            
            # unif = uniform_labels(targets,d n_classes=n_classes)
            # loss_ent = F.kl_div(out_ood, unif)
            loss_ent = loss_ent * 0.00001 #args.lambda_ent
            # print("loss ood Maha_dist:", loss_ent.item())
            loss = loss + loss_ssl + loss_ent
            #out_ood = self.net_cls(out_ood)
        else:
          if self.args.model == 'bert':
              enc_out = self.encoder(inputs['input_ids_ood'], inputs['token_type_ids'], inputs['attention_mask'])
              sequence, pooled = enc_out['last_hidden_state'], enc_out['pooler_output']  # pooled feature
              # out_ood = self.dropout(pooled)
              hidden = self.dropout(pooled)
              out_ood_logits = self.classify(hidden, outcome) # batch_size, num_intents
              out_ood = F.log_softmax(out_ood_logits, dim=1)  # log-probs
              n_classes = out_ood.shape[1]
              #pdb.set_trace()
              unif = uniform_labels(targets, n_classes=n_classes)
              loss_ent = F.kl_div(out_ood, unif)
              loss_ent = loss_ent * 0.0001 #args.lambda_ent
              print("loss ood:", loss_ent.item())
              loss = loss + loss_ssl + loss_ent
              #out_ood = self.net_cls(out_ood)
          if self.args.model == 'roberta':
              enc_out = self.encoder(inputs['input_ids_ood'], inputs['attention_mask'])
              sequence, pooled = enc_out['last_hidden_state'], enc_out['pooler_output']
              # out = self.backbone(x_orig, attention_mask)[0]
              # hidden = sequence[:, 0, :] # take cls token (<s>)
              # hidden = self.dropout(hidden)
              hidden = self.dropout(pooled)
              # hidden = self.dense(hidden)
              # hidden = torch.tanh(hidden)
              # hidden = self.dropout(hidden)
              # batch_s = hidden.shape[0]
              out_ood_logits = self.classify(hidden, outcome) # batch_size, num_intents
              out_ood = F.log_softmax(out_ood_logits, dim=1)  # log-probs
              n_classes = out_ood.shape[1]
              # pdb.set_trace()
              unif = uniform_labels(targets, n_classes=n_classes)
              loss_ent = F.kl_div(out_ood, unif)
              loss_ent = loss_ent * 0.0001 #args.lambda_ent
              print("loss ood:", loss_ent.item())
              loss = loss + loss_ssl + loss_ent
              #out_ood = self.net_cls(out_ood)
        print(loss.item(), loss_ssl.item(), loss_ent.item())
        return output, out_ssl_logits, out_ood_logits, loss