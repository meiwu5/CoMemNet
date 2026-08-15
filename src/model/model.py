import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from model.mlp import MultiLayerPerceptron
from model.TMRB import TMRB

class Basic_Model(nn.Module):
    def __init__(self, args):
        super(Basic_Model, self).__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dropout = args.dropout
        self.activation = nn.GELU()
        self.num_feat = args.emb["num_feat"]
        self.args = args
        self.num_layer = args.emb["num_layer"]

        self.embed_dim = args.emb["adaptive_emb_dim"]
        self.node_dim = args.emb["D^N"]
        self.temp_dim_tid = args.emb["D^D"]
        self.temp_dim_diw = args.emb["D^W"]
        self.output_len = args.emb["output_len"]
        self.tcn_dim = args.tcn["out_channel"]
        self.is_TMRB = args.is_TMRB
        self.is_update = args.is_update
        self.select_k = args.select_k
        self.TMRB_dropout = args.TMRB["dropout"]

        self.node_embedding = nn.init.xavier_uniform_(
                nn.Parameter(torch.empty(1, self.node_dim))
            )
        self.T_i_D_emb = nn.init.xavier_uniform_(nn.Parameter(torch.empty(288, self.temp_dim_tid)))
        self.D_i_W_emb  = nn.init.xavier_uniform_(nn.Parameter(torch.empty(7, self.temp_dim_diw)))
        self.emb_layer_history = nn.Conv2d(in_channels=args.emb["input_dim"]*args.emb["input_len"], out_channels=self.embed_dim, kernel_size=(1, 1), bias=True)
        self.tcn = nn.Conv1d(in_channels=args.tcn["in_channel"], out_channels=args.tcn["out_channel"], kernel_size=args.tcn["kernel_size"], \
            dilation=args.tcn["dilation"], padding=int((args.tcn["kernel_size"]-1)*args.tcn["dilation"]/2))
        self.hidden_dim = self.embed_dim + self.node_dim + args.TMRB["out_channel"]*self.is_TMRB +self.tcn_dim
        self.encoder = nn.Sequential(
            *[MultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)]
        )
        self.projection_head = nn.Conv2d(
            in_channels=self.hidden_dim, out_channels=self.output_len, kernel_size=(1, 1), bias=True
        )
        self.online_backbone = self.encoder
        self.online_projection = self.projection_head
        self.use_target_branch = bool(getattr(args, "use_target_branch", True))
        self.target_backbone = copy.deepcopy(self.online_backbone)
        self.target_projection = copy.deepcopy(self.online_projection)
        for param in self.target_backbone.parameters():
            param.requires_grad = False
        for param in self.target_projection.parameters():
            param.requires_grad = False

        self.momentum = args.momentum
        self.TMRB = TMRB(input_dim=args.TMRB["in_channel"], out_dim=args.TMRB["out_channel"],top_k = args.TMRB["top_k"],TMRB_dropout=self.TMRB_dropout,is_update=self.is_update,select_k = self.select_k).to(self.device)
        self.hidden_states_per_year = {}

    def prepare_inputs(self, history_data):
        batch_size, in_steps, num_nodes, num_channels = history_data.shape
        node_emb = self.node_embedding.expand(size=(num_nodes, *self.node_embedding.shape))
        node_emb = node_emb.expand(size=(batch_size, *node_emb.shape)).transpose(1, 2)

        time_in_day_feat = self.T_i_D_emb[(history_data[:, -1, :, self.num_feat] * 288).long()].to(self.device)
        day_in_week_feat = self.D_i_W_emb[(history_data[:, -1, :, self.num_feat + 1]).long()].to(self.device)

        input_data = history_data[:, :, :, :self.num_feat]
        return input_data, time_in_day_feat, day_in_week_feat, node_emb

    def _build_features(self, history_data, year, update_memory=True, memory_context_year=None):
        batch_size, in_steps, num_nodes, num_features = history_data.shape
        input_data, time_in_day_feat, day_in_week_feat, node_emb = self.prepare_inputs(history_data)

        history_data = history_data.transpose(1, 2).contiguous().view(batch_size, num_nodes, -1).transpose(1, 2).unsqueeze(-1)
        node_emb_list = [node_emb.transpose(1, -1)]
        emb_history = self.emb_layer_history(history_data)
        tcn_emb = self.tcn(emb_history.squeeze(-1))

        tem_emb = torch.cat([time_in_day_feat, day_in_week_feat],dim=-1)
        combined_features = torch.cat([emb_history] + node_emb_list + [tcn_emb.unsqueeze(-1)], dim=1)

        if self.is_TMRB:
            hidden_state = self.TMRB(tem_emb, year, self.hidden_states_per_year,
                                     memory_context_year=memory_context_year)
            if update_memory:
                self.hidden_states_per_year[year] = hidden_state.mean(dim=(0,2))
            combined_features = torch.cat((combined_features,hidden_state.unsqueeze(-1)), dim=1)
        return combined_features

    def forward(self, data, year, return_features=False, memory_context_year=None):
        current_data = data['x']
        combined_features = self._build_features(
            current_data, year, update_memory=self.training,
            memory_context_year=memory_context_year)

        online_features = self.online_backbone(combined_features)
        online_proj = self.online_projection(online_features)
        if return_features:
            return online_proj, online_features
        return online_proj

    def online_branch(self, data, year):
        history_data = data['x'].to(self.device)
        combined_features = self._build_features(history_data, year, update_memory=False)
        with torch.no_grad():
            online_features = self.online_backbone(combined_features)
            online_proj = self.online_projection(online_features)
        return online_proj

    def update_target_network(self):
        if not self.use_target_branch:
            return
        with torch.no_grad():
            for param_o, param_t in zip(self.online_backbone.parameters(), self.target_backbone.parameters()):
                param_t.data = param_t.data * self.momentum + param_o.data * (1. - self.momentum)
            for param_o, param_t in zip(self.online_projection.parameters(), self.target_projection.parameters()):
                param_t.data = param_t.data * self.momentum + param_o.data * (1. - self.momentum)

    def reset_target_network(self):
        self.target_backbone.load_state_dict(self.online_backbone.state_dict())
        self.target_projection.load_state_dict(self.online_projection.state_dict())

    def calculate_similarity(self, online_proj, target_proj):
        batch_size, time_steps, num_nodes, feature_dim = online_proj.shape
        online_proj = online_proj.view(-1, feature_dim)
        target_proj = target_proj.view(-1, feature_dim)
        similarity = F.cosine_similarity(online_proj, target_proj)
        similarity = similarity.view(batch_size, time_steps, num_nodes)

        top_k_values, top_k_indices = torch.topk(similarity, self.top_k, dim=-1)
        return top_k_indices

    def target_branch(self, data, year):
        if not self.use_target_branch:
            return self.online_branch(data, year)
        history_data = data['x'].to(self.device)
        combined_features = self._build_features(history_data, year, update_memory=False)

        with torch.no_grad():
            target_features = self.target_backbone(combined_features)
            target_proj = self.target_projection(target_features)

        return target_proj

    def target_representation(self, data, year):
        history_data = data['x'].to(self.device)
        combined_features = self._build_features(history_data, year, update_memory=False)
        with torch.no_grad():
            if self.use_target_branch:
                return self.target_backbone(combined_features)
            return self.online_backbone(combined_features).detach()

    def contrastive_loss(self, online_proj, target_proj):
        top_k_indices = self.calculate_similarity(online_proj, target_proj)
        return top_k_indices
