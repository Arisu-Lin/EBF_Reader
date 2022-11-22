import argparse
import os
import json
from os.path import join


def process_arguments(args):
    args.checkpoint_path = join(args.checkpoint_path, args.name)
    args.prediction_path = join(args.prediction_path, args.name)
    args.max_query_len = 50
    args.max_doc_len = 512


def save_settings(args):
    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(args.prediction_path, exist_ok=True)
    json.dump(args.__dict__, open(join(args.checkpoint_path, "run_settings.json"), 'w'))


def set_config():
    parser = argparse.ArgumentParser()
    data_path = 'output'

    # Required parameters
    parser.add_argument("--name", type=str, default='default')
    parser.add_argument("--prediction_path", type=str, default=join(data_path, 'submissions'))
    parser.add_argument("--checkpoint_path", type=str, default=join(data_path, 'checkpoints'))
    parser.add_argument("--data_dir", type=str, default='data')
    parser.add_argument("--val_dir", type=str, default='data_2020')

    parser.add_argument("--fp16",  action='store_true')

    parser.add_argument("--ckpt_id", type=int, default=0)
    parser.add_argument("--bert_model", type=str, default='../roberta-base-chinese',
                        help='Currently only support bert-base-uncased and bert-large-uncased')

    # learning and log
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--pre_each_epc", type=int, default=5, help="each epoch predict times")
    parser.add_argument("--epc_start_pre", type=int, default=7, help="begin predict during each epoch")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_bert_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument('--decay', type=float, default=1.0)
    parser.add_argument('--early_stop_epoch', type=int, default=0)    
    parser.add_argument("--verbose_step", default=50, type=int)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--seed", default=1, type=int)

    parser.add_argument("--input_dim", type=int, default=768, help="bert-base=768, bert-large=1024")
    parser.add_argument("--k_v_dim", type=int, default=64, help="transformer d_k and d_v ")
    parser.add_argument("--model_gpu", default='0', type=str, help="device to place model.")
    parser.add_argument('--trained_weight',default=None)

    # loss
    parser.add_argument("--type_lambda", type=float, default=1)
    parser.add_argument("--sp_lambda", type=float, default=5)
    parser.add_argument("--sp_threshold", type=float, default=0.5)
    parser.add_argument('--label_type_num', default=4, type=int)#yes/no/unknown/span
    parser.add_argument("--att_back_lambda", type=float, default=6.7)

    args = parser.parse_args()

    process_arguments(args)
    save_settings(args)

    return args
