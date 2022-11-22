import torch
from torch import nn
from torch.autograd import Variable
from model.transformer.Models import get_pad_mask, Encoder
import torch.nn.functional as F
import numpy as np

class reader1(nn.Module):
    def __init__(self, config, encoder):
        super(reader1, self).__init__()
        self.config = config
        self.encoder = encoder
        self.input_dim = config.input_dim
        self.sp_linear = nn.Linear(self.input_dim, 1)
        self.type_linear = nn.Linear(self.input_dim, config.label_type_num)   # yes/no/ans
        self.query_attn = nn.Linear(self.input_dim, 1)
        self.context_attn = nn.Linear(2*self.input_dim, 1)
        self.transformer = Encoder(d_word_vec=self.input_dim, n_layers=2, n_head=8, d_k=config.k_v_dim, d_v=config.k_v_dim,
            d_model=self.input_dim, d_inner=768, pad_idx=0, dropout=0.1, n_position=512, scale_emb=False)
    def forward(self, batch):
        # query_mapping = batch['query_mapping']
        doc_ids, doc_mask= batch['context_idxs'], batch['context_mask']
        all_mapping = batch['all_mapping']  # (batch_size, 512, max_sent) 
        # context_mapping = batch['context_mapping']
        sent_mapping = batch['sent_mapping']
        qc_out = self.encoder(input_ids=doc_ids, attention_mask=doc_mask)[0]
        # q_weight = self.query_attn(qc_out).suqeeze(2) - 1e30 * (1-context_mapping)
        # q_weight = F.softmax(q_weight, dim=-1)
        # q_rep = torch.matmul(q_weight[:,None], qc_out)
        # q_rep = (qc_out*(query_mapping.unsqueeze(2))).max(1)[0]
        # q_rep = q_rep[:,None].expand_as(qc_out)
        # # print(q_rep.size(),qc_out.size())
        # c_attn = self.context_attn(torch.cat([q_rep,qc_out], dim=-1)).squeeze(2) - 1e30 * (1-context_mapping)
        # c_weight = c_attn.unsqueeze(2)*all_mapping - 1e30 * (1-all_mapping)
        # att = c_weight.sum(1)
        # c_weight = F.softmax(c_weight, dim=-1)
        # sent_rep = torch.matmul(c_weight.permute(0,2,1), qc_out)
        # sp_logits = self.sp_linear(sent_rep).squeeze(2)
        # sp_weight = F.softmax(sp_logits, dim=-1)
        # # print(sp_weight.size(), sent_rep.size())
        # doc_rep = torch.matmul(sp_weight.unsqueeze(1), sent_rep)
        # q = (qc_out*query_mapping.unsqueeze(2)).max(1)[0]

        sp_state = all_mapping.unsqueeze(3) * qc_out.unsqueeze(2)  # N x sent x 512 x 300
        c_flw = sp_state.max(1)[0]
        sent_mask = get_pad_mask(sent_mapping,0)
        #
        qc_trf, att = self.transformer(c_flw,src_mask=sent_mask,  return_attns=True)
        att = att[-1]
        att = torch.mean(att,dim=1)
        att = torch.mean(att,dim=1)
        if batch['ids'][0]==4289:
            np.savez('att.npz', att.detach().cpu().numpy())
        sp_logits = self.sp_linear(qc_trf).squeeze(2)
        # print(doc_rep.size())
        # type_logits = self.type_linear(doc_rep.squeeze(1))
        type_logits = self.type_linear(torch.max(qc_out, dim=1)[0])
        return type_logits, qc_out, sp_logits, att
class reader2(nn.Module):
    def __init__(self, config):
        super(reader2, self).__init__()
        self.config = config
        self.input_dim = config.input_dim
        self.start_linear = nn.Linear(self.input_dim, 1)
        self.end_linear = nn.Linear(self.input_dim, 1)
        self.transformer = Encoder(d_word_vec=self.input_dim, n_layers=2, n_head=8, d_k=config.k_v_dim, d_v=config.k_v_dim,
            d_model=self.input_dim, d_inner=768, pad_idx=0, dropout=0.1, n_position=512, scale_emb=False)
        self.cache_S = 0
        self.cache_mask = None
    def get_output_mask(self, outer):
        # (batch, 512, 512)
        S = outer.size(1)
        if S <= self.cache_S:
            return Variable(self.cache_mask[:S, :S], requires_grad=False)
        self.cache_S = S
       
        np_mask = np.tril(np.triu(np.ones((S, S)), 0), S)
        self.cache_mask = outer.data.new(S, S).copy_(torch.from_numpy(np_mask))
        return Variable(self.cache_mask, requires_grad=False)
    def forward(self, batch, qc_out, sp_logits=None):
        query_mapping = batch['query_mapping']  # (batch, 512) 
        context_mapping = batch['context_mapping']  # context
      
        all_mapping = batch['all_mapping']
        if sp_logits is not None:
           
            sent_att = all_mapping * sp_logits[:, None]
            sent_att = torch.sum(sent_att, dim=-1)
            sent_att = torch.nn.functional.softmax(sent_att, dim=-1)

        else:
            sent_att = None
        # sent_att = None
        mask = get_pad_mask(query_mapping+context_mapping,0)
        output,vasul_att = self.transformer(qc_out, src_mask=mask, sent_att=sent_att, attn_mask=None, return_attns=True)


        start_logits = self.start_linear(output).squeeze(2) - 1e30 * (1 - context_mapping)
        end_logits = self.end_linear(output).squeeze(2) - 1e30 * (1 - context_mapping)
        outer = start_logits[:, :, None] + end_logits[:, None]
       
        span_mask = self.get_output_mask(outer)
        sp_mask = torch.matmul(all_mapping.permute(0,2,1)[:,:,:,None], all_mapping.permute(0,2,1)[:,:,None])
        outer = outer - 1e30 * (1 - span_mask[None].expand_as(outer))
        start_position = outer.max(dim=2)[0].max(dim=1)[1]
        end_position = outer.max(dim=1)[0].max(dim=1)[1]
        outer_ = sp_mask * outer[:,None] - 1e30 * (1 -sp_mask)
        sp_att = outer_.max(dim=-1)[0].max(dim=-1)[0]
      
        if batch['ids'][0]==4289:
          
            l_0 = torch.mean(vasul_att[0][0],dim=0)
            vasul_att = vasul_att[1][0]
            
          
            vasul_att = torch.mean(vasul_att,dim=0)
          
            
            np.savez("a.npz", vasul_att.detach().cpu().numpy())
            np.savez("b.npz", l_0.detach().cpu().numpy())
            np.savez('st.npz', F.softmax(start_logits,dim=-1).detach().cpu().numpy())
            np.savez('ed.npz', F.softmax(end_logits,dim=-1).detach().cpu().numpy())
            np.savez('sp.npz', sp_logits.detach().cpu().numpy())
            np.savez('sp_att.npz', sp_att.detach().cpu().numpy())
           
            print(start_position.data.cpu().numpy()[0],end_position.data.cpu().numpy()[0])

        return start_logits, end_logits, sp_att, start_position, end_position

