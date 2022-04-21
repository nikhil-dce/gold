import os, sys, pdb
import numpy as np
import random
import torch

from tqdm import tqdm as progress_bar
from components.logger import ExperienceLogger
from components.models import BaseModel, IntentModel
from assets.static_vars import device, debug_break, direct_modes

from utils.help import set_seed, setup_gpus, check_directories, prepare_inputs
from utils.process import get_dataloader, check_cache, prepare_features, process_data
from utils.load import load_data, load_tokenizer, load_ontology, load_best_model
from utils.evaluate import (make_clusters, make_projection_matrices, 
  process_diff, process_drop, quantify, run_inference, make_projection_matrices,
  process_nml, make_clusters_pred, make_projection_matrices_and_clusters)
from utils.arguments import solicit_params
from app import augment_features

from utils.get_keywords import get_keywords, masked_dataset
import pdb
def run_train(args, model, datasets, tokenizer, exp_logger):
  train_dataloader = get_dataloader(args, datasets['train'], split='train')
  total_steps = len(train_dataloader) // args.n_epochs
  model.setup_optimizer_scheduler(args.learning_rate, total_steps)

  for epoch_count in range(exp_logger.num_epochs):
    exp_logger.start_epoch(train_dataloader)
    train_metric = ''
    model.train()

    for step, batch in enumerate(train_dataloader):
      inputs, labels = prepare_inputs(batch, model)
      #pdb.set_trace()
      pred, loss = model(inputs, labels)
      exp_logger.tr_loss += loss.item()
      loss.backward()

      if args.verbose:
        train_results = quantify(args, pred.detach(), labels.detach(), exp_logger, "train")
        train_metric = train_results[exp_logger.metric]
      torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
      model.optimizer.step()  # backprop to update the weights
      model.scheduler.step()  # Update learning rate schedule
      model.zero_grad()
      exp_logger.log_train(step, train_metric)
      if args.debug and step >= debug_break*args.log_interval:
        break

    eval_res = run_eval(args, model, datasets, tokenizer, exp_logger)
    if args.do_save and eval_res[exp_logger.metric] >= exp_logger.best_score[exp_logger.metric]:
      exp_logger.best_score = eval_res
      exp_logger.save_best_model(model, tokenizer)
    early_stop = exp_logger.end_epoch()
    if early_stop: break

  return exp_logger.best_score

def run_eval(args, model, datasets, tokenizer, exp_logger, split='dev'):

  dataloader = get_dataloader(args, datasets[split], split)

  if split == 'test':
    if args.version == 'augment':
      model.load_dir = model.save_dir
    model = load_best_model(args, model, device)

  outputs = run_inference(args, model, dataloader, exp_logger, split)
  if args.version == 'baseline' and args.method in ['bert_embed', 'mahalanobis', 'gradient']:
    preloader = get_dataloader(args, datasets['train'], split='train')
    clusters, inv_cov_matrix = make_clusters(args, preloader, model, exp_logger, split)
    outputs = process_diff(args, clusters, inv_cov_matrix, *outputs)
  elif args.version == 'baseline' and args.method == 'mahalanobis_preds':
    preloader = get_dataloader(args, datasets['train'], split='dev')
    clusters, inv_cov_matrix = make_clusters_pred(args, preloader, model, exp_logger, split)
    _, test_targets, exp_logger, test_all_encoder_out = outputs
    new_outputs = test_all_encoder_out, test_targets, exp_logger
    outputs = process_diff(args, clusters, inv_cov_matrix, *new_outputs)
  elif args.version == 'baseline' and args.method == 'mahalanobis_nml':
    # Use `middle` (vectors) for preds like in mahalanobis.
    # Use `hidden` embedding for the projection matrix used in NML.
    # `run_inference` returns: preds/vectors (middle), targets, exp_logger, all_encoder_out (hidden)
    preloader = get_dataloader(args, datasets['train'], split='dev')
    p_parallel, p_bot, clusters, inv_cov_matrix = make_projection_matrices_and_clusters(
      args, preloader, model, exp_logger, split)
    # clusters, inv_cov_matrix = make_clusters(args, preloader, model, exp_logger, split)
    # p_parallel, p_bot = make_projection_matrices(args, preloader, model, exp_logger, split)

    # vectors, test_targets, exp_logger, test_all_encoder_out = outputs  
    
    # process_diff uses vector to compute probs for test examples using mahalanobis.
    probs = process_diff(args, clusters, inv_cov_matrix, *outputs[:-1])[0]
    # process_nml uses all_encoder_out (hidden) to compute the pnml regret.

    new_outputs = probs, outputs[1], outputs[2], outputs[0]
    # probs, targets, exp_logger, testset
    outputs = process_nml(args, p_parallel, p_bot, *new_outputs)

  elif args.version == 'baseline' and args.method == 'dropout':
    outputs = process_drop(args, *outputs, exp_logger)
  elif args.version == 'baseline' and args.method == 'nml':
    # preloader = get_dataloader(args, datasets['train'], split='train')
    preloader = get_dataloader(args, datasets['train'], split='dev')
    p_parallel, p_bot = make_projection_matrices(args, preloader, model, exp_logger, split)
    outputs = process_nml(args, p_parallel, p_bot, *outputs)
    
  results = quantify(args, *outputs, split)
  return results
  
if __name__ == "__main__":
  args = solicit_params()
  args = setup_gpus(args)
  args = check_directories(args)
  set_seed(args)

  cache_results, already_exist = check_cache(args)
  tokenizer = load_tokenizer(args)
  ontology = load_ontology(args)

  
  if already_exist:
    features = cache_results
  else:
    target_data = load_data(args, 'target')
    features = prepare_features(args, target_data, tokenizer, cache_results)
    if args.version == 'augment' and not args.do_eval:
      source_data = load_data(args, 'source')
      features = augment_features(args, source_data, features, cache_results, tokenizer, ontology)
  datasets = process_data(args, features, tokenizer, ontology)

  if args.version == 'augment':
    model = BaseModel(args, ontology, tokenizer).to(device)
  elif args.version == 'baseline':
    model = IntentModel(args, ontology, tokenizer).to(device)
  exp_logger = ExperienceLogger(args, model.save_dir)
  
  if args.do_train:
    best_score = run_train(args, model, datasets, tokenizer, exp_logger)
  if args.do_eval:
    run_eval(args, model, datasets, tokenizer, exp_logger, split='test')
  if args.masker == True:
     ##masker
     #pdb.set_trace()
     train_dataloader = get_dataloader(args, datasets['train'], split='dev')
     masked_dataset = get_keywords(args, train_dataloader)
     
     #pdb.set_trace()
     ##masker
