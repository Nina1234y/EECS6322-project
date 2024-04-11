import torch.nn
from GCN import AGCN 
from Attention import PointWiseFeedForward, TimeAwareMultiHeadAttention
import numpy as np

class AGRAN(torch.nn.Module):
    def __init__(self, num_users, num_pois, dev, max_seq_len, dis_thresh, hidden_units, dropout_rate, time_thresh):
        super(AGRAN, self).__init__()

        self.num_users = num_users
        self.num_pois = num_pois
        self.dev = dev
        self.max_seq_len = max_seq_len
        self.dis_thresh = dis_thresh
        self.hidden_units = hidden_units
        self.dropout_rate = dropout_rate 
        self.time_thresh = time_thresh

        self.item_emb = torch.nn.Embedding(self.num_pois+1, self.hidden_units, padding_idx=0)
        self.item_emb_dropout = torch.nn.Dropout(p=self.dropout_rate)


        #self.gcn = AGCN(input_dim=self.hidden_units,output_dim=self.hidden_units,layer=4, dropout=self.dropout_rate)
        self.gcn = AGCN(input_dim=self.hidden_units, output_dim=self.hidden_units, num_layers=4, drop=self.dropout_rate)

        self.abs_pos_K_emb = torch.nn.Embedding(self.max_seq_len, self.hidden_units)
        self.abs_pos_V_emb = torch.nn.Embedding(self.max_seq_len, self.hidden_units)
        self.time_matrix_K_emb = torch.nn.Embedding(self.time_thresh+1, self.hidden_units)
        self.time_matrix_V_emb = torch.nn.Embedding(self.time_thresh+1, self.hidden_units)

        self.dis_matrix_K_emb = torch.nn.Embedding(self.dis_thresh+1, self.hidden_units)
        self.dis_matrix_V_emb = torch.nn.Embedding(self.dis_thresh+1, self.hidden_units)

        self.abs_pos_K_emb_dropout = torch.nn.Dropout(p=self.dropout_rate)
        self.abs_pos_V_emb_dropout = torch.nn.Dropout(p=self.dropout_rate)
        self.time_matrix_K_dropout = torch.nn.Dropout(p=self.dropout_rate)
        self.time_matrix_V_dropout = torch.nn.Dropout(p=self.dropout_rate)

        self.dis_matrix_K_dropout = torch.nn.Dropout(p=self.dropout_rate)
        self.dis_matrix_V_dropout = torch.nn.Dropout(p=self.dropout_rate)

        self.attention_layernorms = torch.nn.ModuleList()
        self.attention_layers = torch.nn.ModuleList()
        self.forward_layernorms = torch.nn.ModuleList()
        self.forward_layers = torch.nn.ModuleList()

        self.last_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)

        for _ in range(3):
            new_attn_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)

            new_attn_layer = TimeAwareMultiHeadAttention(self.hidden_units,
                                                            2,
                                                            self.dropout_rate,
                                                            self.dev)
            self.attention_layers.append(new_attn_layer)

            new_fwd_layernorm = torch.nn.LayerNorm(self.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)

            new_fwd_layer = PointWiseFeedForward(self.hidden_units, self.dropout_rate)
            self.forward_layers.append(new_fwd_layer)


    def seq2feats(self, user_ids, log_seqs, time_matrices, dis_matrices, item_embs):

        seqs = item_embs[torch.LongTensor(log_seqs).to(self.dev),:]
        seqs *= item_embs.shape[1] ** 0.5

        seqs = self.item_emb_dropout(seqs)

        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        positions = torch.LongTensor(positions).to(self.dev)
        abs_pos_K = self.abs_pos_K_emb(positions)
        abs_pos_V = self.abs_pos_V_emb(positions)
        abs_pos_K = self.abs_pos_K_emb_dropout(abs_pos_K)
        abs_pos_V = self.abs_pos_V_emb_dropout(abs_pos_V)

        time_matrices = torch.LongTensor(time_matrices).to(self.dev)
        time_matrix_K = self.time_matrix_K_emb(time_matrices)
        time_matrix_V = self.time_matrix_V_emb(time_matrices)
        time_matrix_K = self.time_matrix_K_dropout(time_matrix_K)
        time_matrix_V = self.time_matrix_V_dropout(time_matrix_V)

        dis_matrices = torch.LongTensor(dis_matrices).to(self.dev)
        dis_matrix_K = self.dis_matrix_K_emb(dis_matrices)
        dis_matrix_V = self.dis_matrix_V_emb(dis_matrices)
        dis_matrix_K = self.dis_matrix_K_dropout(dis_matrix_K)
        dis_matrix_V = self.dis_matrix_V_dropout(dis_matrix_V)

        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1)

        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))

        for i in range(len(self.attention_layers)):
            Q = self.attention_layernorms[i](seqs)
            mha_outputs = self.attention_layers[i](Q, seqs,
                                            timeline_mask, attention_mask,
                                            time_matrix_K, time_matrix_V,
                                            dis_matrix_K,dis_matrix_V,
                                            abs_pos_K, abs_pos_V)
            seqs = Q + mha_outputs

            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs)
            seqs *=  ~timeline_mask.unsqueeze(-1)

        log_feats = self.last_layernorm(seqs)

        return log_feats

    def forward(self, user_ids, log_seqs, time_matrices,dis_matrices, pos_seqs, neg_seqs):
        #item_embs,support = self.gcn(self.item_emb)
        item_embs, support = self.gcn(self.item_emb, self.dev)

        self.item_embs = item_embs
        log_feats = self.seq2feats(user_ids, log_seqs, time_matrices, dis_matrices, item_embs)

        pos_embs = item_embs[torch.LongTensor(pos_seqs).to(self.dev),:]
        neg_embs = item_embs[torch.LongTensor(neg_seqs).to(self.dev),:]

        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)

        fin_logits = log_feats.matmul(item_embs.transpose(0,1))
        fin_logits = fin_logits.reshape(-1,fin_logits.shape[-1])

        return pos_logits, neg_logits,fin_logits, self.item_embs[0],support

    def predict(self, user_ids, log_seqs, time_matrices, dis_matrices, item_indices):
        # item_embs,support = self.gcn(self.item_emb)
        item_embs, support = self.gcn(self.item_emb, self.dev)

        log_feats = self.seq2feats(user_ids, log_seqs, time_matrices, dis_matrices, item_embs)

        final_feat = log_feats[:, -1, :]

        item_emb = item_embs

        logits = final_feat.matmul(item_emb.transpose(0,1))

        return logits,item_indices
