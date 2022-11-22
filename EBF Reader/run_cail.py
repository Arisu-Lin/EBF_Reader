import faulthandler
faulthandler.enable()
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
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)




def dispatch(context_encoding, context_mask, batch, device):
    batch['context_encoding'] = context_encoding.cuda(device)
    batch['context_mask'] = context_mask.float().cuda(device)
    return batch


def compute_loss(batch, start_logits, end_logits, type_logits, sp_logits, start_position, end_position):
    
    loss1 = criterion(start_logits, batch['y1']) + criterion(end_logits, batch['y2'])
    loss2 = args.type_lambda * criterion(type_logits, batch['q_type'])

    sent_num_in_batch = batch["sent_mapping"].sum()
    loss3 = args.sp_lambda * sp_loss_fct(sp_logits.view(-1), batch['is_support'].float().view(-1)).sum() / sent_num_in_batch
    loss = loss1 + loss2 + loss3
    return loss, loss1, loss2, loss3



import json

@torch.no_grad()
def predict(model1, model2, dataloader, example_dict, feature_dict, prediction_file, need_sp_logit_file=False):

    model1.eval()
    model2.eval()
    answer_dict = {}
    sp_dict = {}
    dataloader.refresh()
    total_test_loss = [0] * 4

    for batch in tqdm(dataloader,ncols=80):

        batch['context_mask'] = batch['context_mask'].float()
        batch['all_mapping'] = torch.Tensor(batch['sent_st_ed'].size(0), batch['context_idxs'].size(1),
                                            batch['sent_st_ed'].size(1)).to('cuda')
        batch['all_mapping'].zero_();
        for i in range(batch['sent_st_ed'].size(0)):
            for j in range(batch['sent_st_ed'].size(1)):
                batch['all_mapping'][i, int(batch['sent_st_ed'][i, j, 0]):int(batch['sent_st_ed'][i, j, 1]) + 1, j] = 1
        type_logits, qc_out, sp_logits, att = model1(batch)
        start_logits, end_logits, sp_att, start_position, end_position = model2(batch, qc_out, sp_logits)
        loss_list = compute_loss(batch, start_logits, end_logits, type_logits, sp_logits, start_position, end_position)

        for i, l in enumerate(loss_list):
            if not isinstance(l, int):
                total_test_loss[i] += l.item()


        answer_dict_ = convert_to_tokens(example_dict, feature_dict, batch['ids'], start_position.data.cpu().numpy().tolist(),
                                         end_position.data.cpu().numpy().tolist(), np.argmax(type_logits.data.cpu().numpy(), 1))
        answer_dict.update(answer_dict_)

        predict_support_np = torch.sigmoid(sp_logits).data.cpu().numpy()
        for i in range(predict_support_np.shape[0]):
            cur_sp_pred = []
            cur_id = batch['ids'][i]

            cur_sp_logit_pred = []  # for sp logit output
            for j in range(predict_support_np.shape[1]):
                if j >= len(example_dict[cur_id].sent_names):
                    break
                if need_sp_logit_file:
                    temp_title, temp_id = example_dict[cur_id].sent_names[j]
                    cur_sp_logit_pred.append((temp_title, temp_id, predict_support_np[i, j]))
                if predict_support_np[i, j] > args.sp_threshold:
                    cur_sp_pred.append(example_dict[cur_id].sent_names[j])
            sp_dict.update({cur_id: cur_sp_pred})

    new_answer_dict={}
    for key,value in answer_dict.items():
        new_answer_dict[key]=value.replace(" ","")
    prediction = {'answer': new_answer_dict, 'sp': sp_dict}
    with open(prediction_file, 'w',encoding='utf8') as f:
        json.dump(prediction, f,indent=4,ensure_ascii=False)

    for i, l in enumerate(total_test_loss):
        print("Test Loss{}: {}".format(i, l / len(dataloader)))
    test_loss_record.append(sum(total_test_loss[:3]) / len(dataloader))


def train_epoch(data_loader, model1, model2, predict_during_train=False, best_join_f1=0):
    model1.train()
    model2.train()
    pbar = tqdm(total=len(data_loader),ncols=80)
    epoch_len = len(data_loader)
    step_count = 0
    predict_step = epoch_len // args.pre_each_epc
    jf1_tmp = []
    while not data_loader.empty():
        step_count += 1
        batch = next(iter(data_loader))
        batch['context_mask'] = batch['context_mask'].float()
        batch['all_mapping'] = torch.Tensor(batch['sent_st_ed'].size(0), batch['context_idxs'].size(1), batch['sent_st_ed'].size(1)).to('cuda')
        batch['all_mapping'].zero_();
        for i in range(batch['sent_st_ed'].size(0)):
            for j in range(batch['sent_st_ed'].size(1)):
                batch['all_mapping'][i,int(batch['sent_st_ed'][i,j,0]):int(batch['sent_st_ed'][i,j,1])+1,j] = 1
        train_bh(model1,model2, batch)
        del batch
        if predict_during_train and (step_count % predict_step == 0):
            predict(model1, model2, eval_dataset, dev_example_dict, dev_feature_dict,
                     join(args.prediction_path, 'pred_seed_{}_epoch_{}_{}.json'.format(args.seed, epc, step_count)))
            jf1=eval(join(args.prediction_path, 'pred_seed_{}_epoch_{}_{}.json'.format(args.seed, epc, step_count)),join(args.val_dir, 'dev.json'))
            if jf1>best_join_f1:
                model1_to_save = model1.module if hasattr(model1, 'module') else model1
                model2_to_save = model2.module if hasattr(model2, 'module') else model2
                torch.save(model1_to_save.state_dict(), join(args.checkpoint_path, 'best_ckpt_model1.pkl'))
                torch.save(model2_to_save.state_dict(), join(args.checkpoint_path, 'best_ckpt_model2.pkl'))
                best_join_f1 = jf1
            model1.train()
            model2.train()
        pbar.update(1)

    predict(model1, model2, eval_dataset, dev_example_dict, dev_feature_dict,
             join(args.prediction_path, 'pred_seed_{}_epoch_{}_99999.json'.format(args.seed, epc)))
    jf1=eval(join(args.prediction_path, 'pred_seed_{}_epoch_{}_99999.json'.format(args.seed, epc)),join(args.val_dir, 'dev.json'))
    if jf1>best_join_f1:
        model1_to_save = model1.module if hasattr(model1, 'module') else model1
        model2_to_save = model2.module if hasattr(model2, 'module') else model2
        torch.save(model1_to_save.state_dict(), join(args.checkpoint_path, 'best_ckpt_model1.pkl'))
        torch.save(model2_to_save.state_dict(), join(args.checkpoint_path, 'best_ckpt_model2.pkl'))
        best_join_f1 = jf1
    return best_join_f1
def train_bh(model1,model2,batch):
    global global_step, total_train_loss

    type_logits, qc_out, sp_logits, att = model1(batch)
    start_logits, end_logits, sp_att, start_position, end_position = model2(batch, qc_out, sp_logits)
    loss1 = criterion(start_logits, batch['y1']) + criterion(end_logits, batch['y2'])
    sp_att = sp_att.detach()
    loss2 = args.type_lambda * criterion(type_logits, batch['q_type'])
    sent_num_in_batch = batch["sent_mapping"].sum()
    # print(sent_num_in_batch, batch["sent_mapping"].size())
    # exit()
    loss3 = args.sp_lambda * sp_loss_fct(sp_logits.view(-1),
                                         batch['is_support'].float().view(-1)).sum() / sent_num_in_batch
    # loss_s = args.att_back_lambda * sp_loss_fct(att.view(-1), sp_att.view(-1)).sum() / sent_num_in_batch
    loss = loss1 + loss2 + loss3 
    if args.gradient_accumulation_steps > 1:
        loss = loss / args.gradient_accumulation_steps

    loss.backward()
    if (global_step + 1) % args.gradient_accumulation_steps == 0:
        optimizer1.step()
        optimizer2.step()
        scheduler1.step()
        scheduler2.step()
        optimizer1.zero_grad()
        optimizer2.zero_grad()
    global_step += 1

    for _ in range(1):

        type_logits, qc_out, sp_logits, att = model1(batch)
        loss_tp = args.type_lambda * criterion(type_logits, batch['q_type'])
        sent_num_in_batch = batch["sent_mapping"].sum()
        temp = 5
        # loss_s = args.att_back_lambda * sp_loss_fct(att.view(-1), sp_att.view(-1)).sum() / sent_num_in_batch
        loss_s = args.att_back_lambda *kl_loss(F.log_softmax(att/temp,dim=-1), F.softmax(sp_att/temp, dim=-1))
        loss_sp = args.sp_lambda * sp_loss_fct(sp_logits.view(-1),
                                               batch['is_support'].float().view(-1)).sum() / sent_num_in_batch
        loss_1 = loss_tp + loss_sp + loss_s
        if args.gradient_accumulation_steps > 1:
            loss_1 = loss_1 / args.gradient_accumulation_steps

        loss_1.backward()

        if (global_step + 1) % args.gradient_accumulation_steps == 0:
            optimizer1.step()
            scheduler1.step()
            optimizer1.zero_grad()

        global_step += 1


def parm_count(model):
    total_params = sum(p.numel() for p in model.parameters()) 
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) 
    print(f'{total_trainable_params:,} training parameters.')


if __name__ == "__main__":
    scaler = GradScaler()
    parser = argparse.ArgumentParser()
    args = set_config()

    args.n_gpu = torch.cuda.device_count()

    set_seed(args)# Added here for reproductibility

    helper = DataHelper(gz=True, config=args)
    args.n_type = helper.n_type  # 2

    # Set datasets
    Full_Loader = helper.train_loader
    dev_example_dict = helper.dev_example_dict
    dev_feature_dict = helper.dev_feature_dict
    eval_dataset = helper.dev_loader
    

    roberta_config = AutoConfig.from_pretrained(args.bert_model)
    encoder = AutoModel.from_pretrained(args.bert_model)
    
    args.input_dim=roberta_config.hidden_size
    model1 = reader1(config=args, encoder=encoder)
    model2 = reader2(config=args)
    # parm_count(model1)
    # parm_count(model2)
    model1.to('cuda')
    model2.to('cuda')
    # model1.load_state_dict(torch.load(join(args.checkpoint_path, "ckpt_model1_seed_78_epoch_20_99999.pth")))
    # model2.load_state_dict(torch.load(join(args.checkpoint_path, "ckpt_model2_seed_78_epoch_20_99999.pth")))
    # Initialize optimizer and criterions
    lr = args.lr
    t_total = len(Full_Loader) * 2 * args.epochs  // args.gradient_accumulation_steps

    warmup_steps = 0.1 * t_total
    # print(warmup_steps)
    optimizer1 = AdamW(model1.parameters(), lr=lr, eps=1e-8)
    optimizer2 = AdamW(model2.parameters(), lr=lr, eps=1e-8)
    
    scheduler1 = get_linear_schedule_with_warmup(optimizer1, num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)
    scheduler2 = get_linear_schedule_with_warmup(optimizer2, num_warmup_steps=warmup_steps,
                                                num_training_steps=t_total)
    criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=IGNORE_INDEX)  
    binary_criterion = nn.BCEWithLogitsLoss(reduction='mean')  
    sp_loss_fct = nn.BCEWithLogitsLoss(reduction='none')  
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    model1 = torch.nn.DataParallel(model1)
    model1.train()
    model2 = torch.nn.DataParallel(model2)
    model2.train()
    # Training
    global_step = epc = 0
    total_train_loss = [0] * 5
    test_loss_record = []
    VERBOSE_STEP = args.verbose_step
    best_join_f1 = 0
    while True:
        if epc == args.epochs:  
            print('best join f1:',best_join_f1)
            del model1,model2,Full_Loader,eval_dataset
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
            exit(0)
        epc += 1
        print("epoch:",epc)
        Loader = Full_Loader
        Loader.refresh()
        if epc<args.epc_start_pre:
            best_join_f1=train_epoch(Loader, model1, model2,best_join_f1=best_join_f1)

        else :
            best_join_f1=train_epoch(Loader, model1, model2, predict_during_train=True,best_join_f1=best_join_f1)
