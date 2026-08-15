import torch
import torch.nn as nn
import torch.nn.functional as F

class GatedUpdateCell(nn.Module):
    def __init__(self, input_dim, hidden_dim,TMRB_dropout):
        super(GatedUpdateCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.input_to_hidden = nn.Linear(input_dim, hidden_dim)

        self.W_r = nn.Linear(hidden_dim*2, hidden_dim)
        self.W_z = nn.Linear(hidden_dim*2, hidden_dim)
        self.W_t = nn.Linear(hidden_dim*2, hidden_dim)
        self.dropout = nn.Dropout(p=TMRB_dropout)

    def forward(self, x, h_prev):
        if x.size(-1) != self.hidden_dim:
            x = self.input_to_hidden(x)
        combined = self.dropout(torch.cat((x, h_prev), dim=-1))
        r_t = torch.sigmoid(self.W_r(combined))
        z_t = torch.sigmoid(self.W_z(combined))
        h_t = torch.tanh(self.W_t(torch.cat((x, h_prev * r_t), dim=-1)))
        h_next = z_t * h_t + (1 - z_t) * x
        return h_next

class TMRB(nn.Module):
    def __init__(self, input_dim, out_dim,top_k, TMRB_dropout,is_update,select_k,device="cuda:0"):
        super(TMRB,self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        self.device = device
        self.TMRB_dropout = TMRB_dropout
        self.is_update = is_update
        self.GatedUpdateCell = GatedUpdateCell(out_dim, out_dim,self.TMRB_dropout).to(self.device)
        self.top_k = top_k
        self.select_k = select_k
        self.mlp = nn.Linear(self.top_k * self.input_dim, self.out_dim)
        self.init_hidden = nn.Parameter(
    nn.init.xavier_uniform_(torch.empty(input_dim, 1)))
        self.dorpout = nn.Dropout(TMRB_dropout)

    def forward(self, tem_emb, year, hidden_states_per_year, memory_context_year=None):
        B, N, D = tem_emb.shape
        tem_emb = tem_emb.transpose(1,-1)

        context_year = year - 1 if memory_context_year is None else memory_context_year
        if context_year in hidden_states_per_year.keys():
            prev_hidden = hidden_states_per_year[context_year]
            prev_hidden = prev_hidden.expand(size = (N,*prev_hidden.shape)).transpose(0,1)
        else:
            prev_hidden = self.init_hidden
            prev_hidden = prev_hidden.expand(size = (N,*prev_hidden.shape)).transpose(0,1)

        prev_hidden = prev_hidden.expand(size = (B,*prev_hidden.shape))
        prev_hidden = prev_hidden.squeeze(-1)

        if self.select_k:
            time_step_diff = torch.abs(tem_emb - prev_hidden)
            _, top_k_indices = torch.topk(time_step_diff, k=self.top_k, dim=-1, largest=True, sorted=False)
            top_k_features = torch.gather(tem_emb, dim=2, index=top_k_indices)


        else:
            top_k_indices = torch.randint(0, N, (B, self.top_k), device=tem_emb.device)
            top_k_features = torch.gather(tem_emb, dim=2, index=top_k_indices.unsqueeze(-1).expand(-1, -1, D))

        top_k_features = top_k_features.view(B, -1)
        time_step_input = self.mlp(top_k_features)

        if self.is_update:
            prev_hidden_avg = torch.mean(prev_hidden, dim=2)
            prev_hidden_avg = prev_hidden_avg.view(B, -1)
            updated_hidden_pool = self.GatedUpdateCell(time_step_input,prev_hidden_avg)
        else:
            updated_hidden_pool = time_step_input

        updated_hidden_pool = updated_hidden_pool.unsqueeze(2).expand(B, self.out_dim, N)
        return updated_hidden_pool