import argparse
import os
from os.path import join
from tqdm import tqdm
from transformers import AutoModel, AutoConfig
from evaluate import eval
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from model.modeling import *
from tools.utils import convert_to_tokens
from tools.data_iterator_pack import IGNORE_INDEX
import numpy as np
import queue
import random
from config import set_config
from tools.data_helper import DataHelper
from data_process import InputFeatures,Example
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)
        
parser = argparse.ArgumentParser()
args = set_config()

args.n_gpu = torch.cuda.device_count()

set_seed(args)# Added here for reproductibility
roberta_config = AutoConfig.from_pretrained(args.bert_model)
encoder = AutoModel.from_pretrained(args.bert_model)

args.input_dim=roberta_config.hidden_size
helper = DataHelper(gz=True, config=args)
args.n_type = helper.n_type  # 2
test_example_dict = helper.test_example_dict
test_feature_dict = helper.test_feature_dict
test_dataset = helper.test_loader
model1 = reader1(config=args, encoder=encoder)
model2 = reader2(config=args)
model1.to('cuda')
model2.to('cuda')
print("***************************************test***************************************")
model1.load_state_dict(torch.load(join(args.checkpoint_path, 'best_ckpt_model1.pkl')))
model2.load_state_dict(torch.load(join(args.checkpoint_path, 'best_ckpt_model2.pkl')))
model1 = torch.nn.DataParallel(model1)
model2 = torch.nn.DataParallel(model2)
predict(model1, model2, test_dataset, test_example_dict, test_feature_dict,join(args.prediction_path, 'test_pred.json'))
_ = eval(join(args.prediction_path, 'test_pred.json'),join(args.val_dir, 'test.json'))