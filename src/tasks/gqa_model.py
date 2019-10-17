# coding=utf-8
# Copyleft 2019 project LXRT.

import torch
import torch.nn as nn
import torch.jit as jit
from torch import Tensor  # noqa: F401
import torch.nn.functional as F

from param import args
from lxrt.entry import LXRTEncoder
from lxrt.modeling import BertLayerNorm, GeLU

# Max length including <bos> and <eos>
MAX_GQA_LENGTH = 20


class LanguageConditionedGraph(jit.ScriptModule):
    __constants__ = ['time_steps', 'inner_dim']

    def __init__(self, time_steps, inner_dim, vis_dim, lang_dim):
        super(LanguageConditionedGraph, self).__init__()
        # self.time_steps = time_steps
        self.inner_dim = inner_dim
        self.lang_dim = lang_dim
        self.vis_dim = vis_dim
        self.time_steps = time_steps
        # self.current_time_step = 0
        self.starting_state = nn.Parameter(torch.randn(1, 1, self.inner_dim))

        self.vis_reduction = nn.Linear(vis_dim, inner_dim)
        self.vis_dim = inner_dim
        # Language input layers
        self.w1 = nn.Linear(self.lang_dim, 1)
        self.w2 = nn.ModuleList([nn.Linear(self.lang_dim, self.lang_dim)
                                 for _ in range(self.time_steps)])
        self.w3 = nn.Linear(self.lang_dim, self.lang_dim)

        self.w4 = nn.Linear(self.vis_dim, self.inner_dim)
        self.w5 = nn.Linear(self.inner_dim, self.inner_dim)
        combined_dim = self.vis_dim + 2 * self.inner_dim
        self.w6 = nn.Linear(combined_dim, self.inner_dim)
        self.w7 = nn.Linear(combined_dim, self.inner_dim)
        self.w8 = nn.Linear(self.lang_dim, self.inner_dim)
        self.w9 = nn.Linear(combined_dim, self.inner_dim)
        self.w10 = nn.Linear(self.vis_dim, self.inner_dim)
        self.w11 = nn.Linear(2 * self.inner_dim, self.inner_dim)
        self.w12 = nn.Linear(self.vis_dim + self.inner_dim, self.inner_dim)
        self.relu = nn.ELU()

        self.read_drop = nn.Dropout(0.15)

    @jit.script_method
    def forward(self, input, hid, query, vis_mask, lang_mask):
        # type: (Tensor, Tensor, Tensor, Tensor, Tensor) -> Tensor
        # lengths = (1 - lang_mask).sum(-1) - 1
        # lengths = lengths.unsqueeze(0).expand(1, hid.size(-2), hid.size(-1))
        # query = hid.gather(0, lengths)

        state = self.starting_state.expand(input.size(0), input.size(1), -1)

        expanded_mask = vis_mask.expand(
            vis_mask.size(0), vis_mask.size(-1), -1)
        expanded_mask = (
            expanded_mask.transpose(-2, -1) + expanded_mask).clamp(0, 1)
        expanded_mask = expanded_mask.bool()
        lang_mask = lang_mask.unsqueeze(-1)

        input = F.normalize(input, dim=-1)
        input = self.vis_reduction(input)

        prob = 1.0 if self.training else 0.85

        mask = torch.rand(input.size(), device='cuda').le(prob)
        mask = mask.float() / prob

        dim = torch.tensor(state.size(-1), device=state.device).float()
        sqrt_dim = torch.sqrt(dim)

        for w2 in self.w2:
            q_composition = w2(self.relu(self.w3(query)))
            hid_infused = hid * q_composition
            hid_infused = self.w1(hid_infused)
            hid_infused = hid_infused.masked_fill(lang_mask, -1e18)
            alpha = torch.softmax(hid_infused, dim=0)
            query_weighted = (alpha * hid).sum(1).unsqueeze(1)

            state = state * mask
            state_input = (self.w4(self.read_drop(input)) *
                           self.w5(self.read_drop(state)))
            x_tilde = torch.cat([input, state, state_input], dim=-1)

            right_dot = self.w7(x_tilde) * self.w8(query_weighted)
            left_dot = self.w6(x_tilde)

            inner_dot = torch.matmul(left_dot, right_dot.transpose(1, 2))
            inner_dot = inner_dot / sqrt_dim
            inner_dot = inner_dot.masked_fill(expanded_mask, -1e18)

            attn = torch.softmax(inner_dot, dim=-1)
            bottleneck = self.w9(x_tilde) * self.w10(query_weighted)
            # B, N, C = bottleneck.size()
            # bottleneck = bottleneck.unsqueeze(1).expand(B, N, N, C)
            # messages = (attn * bottleneck).sum(1)
            messages = torch.matmul(attn, bottleneck)
            out = torch.cat([state, messages], dim=-1)
            state = self.w11(out)
        state = self.w12(torch.cat([input, state], dim=-1))
        return state


class OriginalVQAClassificationHead(nn.Module):
    def __init__(self, lang_size, ctx_size, output_space):
        super(OriginalVQAClassificationHead, self).__init__()
        # vis_size = cfg.MODEL.BACKBONE.OUTPUT_CHANNELS + additional_coors
        # bidirectional = cfg.MODEL.LANG.BIDIRECTIONAL
        # lang_size = cfg.MODEL.LANG.HID_SIZE * (1 + bidirectional)
        # ctx_size = cfg.MODEL.GRAPH.SIZE

        self.loss = nn.CrossEntropyLoss()
        self.lang_mapper = nn.Linear(lang_size, ctx_size)
        self.w13 = nn.Linear(ctx_size, 1)
        self.w14 = nn.Linear(lang_size, ctx_size)
        self.w16 = nn.Linear(2 * ctx_size + lang_size, 512)
        self.w15 = nn.Linear(512, output_space)
        self.relu = nn.ELU()

        self.out_drop1 = nn.Dropout(0.15)
        self.out_drop2 = nn.Dropout(0.15)
        self.output = nn.Sequential(self.out_drop1,
                                    self.w16,
                                    self.relu,
                                    self.out_drop2,
                                    self.w15)

    def forward(self, query, state, vis_mask):
        # lengths = (1 - lang_mask).sum(-1) - 1
        # lengths = lengths.unsqueeze(0).expand(1, hid.size(-2), hid.size(-1))
        # query = hid.gather(0, lengths).transpose(0, 1)
        vis_mask = vis_mask.bool()
        comb = state * self.w14(query)
        comb = F.normalize(comb, dim=-1)
        comb = self.w13(comb)
        comb = comb.masked_fill(vis_mask.transpose(-2, -1), -1e18)

        attn = torch.softmax(comb, dim=1)
        weighted = (attn * state).sum(1)

        query = self.lang_mapper(query)
        prod = weighted * query.squeeze()
        out = torch.cat([weighted, query.squeeze(), prod], dim=-1)
        # out = self.w15(self.relu(self.w16(out)))
        out = self.output(out)
        return out


class GQAModel(nn.Module):
    def __init__(self, num_answers):
        super().__init__()
        self.lxrt_encoder = LXRTEncoder(
            args,
            max_seq_length=MAX_GQA_LENGTH
        )
        hid_dim = self.lxrt_encoder.dim
        self.graph = LanguageConditionedGraph(4, hid_dim, hid_dim, hid_dim)
        self.output = OriginalVQAClassificationHead(
            hid_dim, hid_dim, num_answers)
        self.logit_fc = nn.Sequential(
            nn.Linear(hid_dim, hid_dim * 2),
            GeLU(),
            BertLayerNorm(hid_dim * 2, eps=1e-12),
            nn.Linear(hid_dim * 2, num_answers)
        )
        self.logit_fc.apply(self.lxrt_encoder.model.init_bert_weights)

    def forward(self, feat, pos, sent):
        """
        b -- batch_size, o -- object_number, f -- visual_feature_size

        :param feat: (b, o, f)
        :param pos:  (b, o, 4)
        :param sent: (b,) Type -- list of string
        :param leng: (b,) Type -- int numpy array
        :return: (b, num_answer) The logit of each answers.
        """
        (feats, _), lang_mask = self.lxrt_encoder(sent, (feat, pos))
        lengths = lang_mask.sum(1).unsqueeze(-1)
        lang_mask = 1 - lang_mask

        lang_feats, vis_feats = feats
        vis_mask = torch.zeros(vis_feats.size(0), 1, vis_feats.size(1),
                               device=vis_feats.device)
        lang_mask = lang_mask.bool()
        mask = lang_mask.unsqueeze(-1).expand(lang_feats.size())
        lang_feats = lang_feats.masked_fill(mask.bool(), 1e-32)
        query = lang_feats.sum(1) / lengths
        query = query.unsqueeze(1)

        x = self.graph(vis_feats, lang_feats, query, vis_mask, lang_mask)
        logit = self.output(query, x, vis_mask)
        # logit = self.logit_fc(x)

        return logit
