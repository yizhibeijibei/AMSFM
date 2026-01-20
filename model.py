import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================
# 损失函数 + BudgetLoss 集成
# ==========================================================
def get_loss(prediction, ground_truth, base_price, mask, batch_size,
             alpha, gamma=0.0, budget_loss=None):
    device = prediction.device
    all_one = torch.ones(batch_size, 1, dtype=torch.float32).to(device)

    # 转收益率 r = (p - p0) / p0
    return_ratio = torch.div(torch.sub(prediction, base_price), base_price)

    # 回归损失（带 mask 的 MSE）
    reg_loss = F.mse_loss(return_ratio * mask, ground_truth * mask)

    # 排序损失（pairwise rank loss）
    pre_pw_dif = torch.sub(
        return_ratio @ all_one.t(),
        all_one @ return_ratio.t()
    )
    gt_pw_dif = torch.sub(
        all_one @ ground_truth.t(),
        ground_truth @ all_one.t()
    )
    mask_pw = mask @ mask.t()
    rank_loss = torch.mean(
        F.relu(pre_pw_dif * gt_pw_dif * mask_pw)
    )

    # 基础 loss
    loss = reg_loss + alpha * rank_loss

    # 可选：BudgetLoss（窗口选择的分布约束）
    budget_loss_val = torch.tensor(0.0, device=device)
    if (budget_loss is not None) and (gamma > 0.0):
        budget_loss_val = budget_loss
        loss = loss + gamma * budget_loss_val

    return loss, reg_loss, rank_loss, budget_loss_val, return_ratio


# ==========================================================
# RWKV Components
# ==========================================================
class RWKV_TimeMix(nn.Module):
    def __init__(self, layer_id, n_embd, n_attn, n_head, ctx_len):
        super().__init__()
        assert n_attn % n_head == 0
        self.layer_id = layer_id
        self.ctx_len = ctx_len
        self.n_head = n_head
        self.head_size = n_attn // n_head

        with torch.no_grad():
            ww = torch.ones(n_head, ctx_len)
            curve = torch.tensor([-(ctx_len - 1 - i) for i in range(ctx_len)])
            for h in range(n_head):
                if h < n_head - 1:
                    decay_speed = math.pow(ctx_len, -(h + 1) / (n_head - 1))
                else:
                    decay_speed = 0.0
                ww[h] = torch.exp(curve * decay_speed)
        self.time_w = nn.Parameter(ww)

        self.time_alpha = nn.Parameter(torch.ones(self.n_head, 1, ctx_len))
        self.time_beta = nn.Parameter(torch.ones(self.n_head, ctx_len, 1))
        self.time_gamma = nn.Parameter(torch.ones(ctx_len, 1))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        self.key = nn.Linear(n_embd, n_attn)
        self.value = nn.Linear(n_embd, n_attn)
        self.receptance = nn.Linear(n_embd, n_attn)
        self.output = nn.Linear(n_attn, n_embd)

        self.key.scale_init = 0
        self.receptance.scale_init = 0
        self.output.scale_init = 0

    def forward(self, x):
        B, T, C = x.size()
        TT = self.ctx_len

        w = F.pad(self.time_w, (0, TT))
        w = torch.tile(w, [TT])
        w = w[:, :-TT].reshape(-1, TT, 2 * TT - 1)
        w = w[:, :, TT - 1:]
        w = w[:, :T, :T] * self.time_alpha[:, :, :T] * self.time_beta[:, :T, :]

        x = torch.cat([self.time_shift(x[:, :, :C // 2]), x[:, :, C // 2:]], dim=-1)

        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)

        k = torch.clamp(k, max=30, min=-60)
        k = torch.exp(k)
        sum_k = torch.cumsum(k, dim=1)

        kv = (k * v).view(B, T, self.n_head, self.head_size)
        wkv = (torch.einsum('htu,buhc->bthc', w, kv)).contiguous().view(B, T, -1)

        rwkv = torch.sigmoid(r) * wkv / sum_k
        rwkv = self.output(rwkv)

        return rwkv * self.time_gamma[:T, :]


class RWKV_ChannelMix(nn.Module):
    def __init__(self, layer_id, n_embd, n_ffn, hidden_sz, n_attn, n_head, ctx_len):
        super().__init__()
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        hidden_sz = 5 * n_ffn // 2
        self.key = nn.Linear(n_embd, hidden_sz)
        self.value = nn.Linear(n_embd, hidden_sz)
        self.weight = nn.Linear(hidden_sz, n_embd)
        self.receptance = nn.Linear(n_embd, n_embd)

        self.receptance.scale_init = 0
        self.weight.scale_init = 0

    def forward(self, x):
        B, T, C = x.size()
        x = torch.cat([self.time_shift(x[:, :, :C // 2]), x[:, :, C // 2:]], dim=-1)
        k = self.key(x)
        v = self.value(x)
        r = self.receptance(x)

        wkv = self.weight(F.mish(k) * v)
        rwkv = torch.sigmoid(r) * wkv
        return rwkv


class Block(nn.Module):
    def __init__(self, n_embd, n_attn, n_head, ctx_len, n_ffn, hidden_sz, model_type="RWKV", layer_id=1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = RWKV_TimeMix(layer_id, n_embd, n_attn, n_head, ctx_len)
        self.mlp = RWKV_ChannelMix(layer_id, n_embd, n_ffn, hidden_sz, n_attn, n_head, ctx_len)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class IEBlock(nn.Module):
    def __init__(self, time_dim, channel_dim, hid_dim=None, n_head=1, layer_id=1):
        super(IEBlock, self).__init__()
        n_embd = channel_dim
        n_attn = channel_dim
        ctx_len = time_dim
        n_ffn = hid_dim if hid_dim is not None else channel_dim
        hidden_sz = n_ffn

        self.block = Block(
            n_embd=n_embd, n_attn=n_attn, n_head=n_head, ctx_len=ctx_len,
            n_ffn=n_ffn, hidden_sz=hidden_sz, model_type="RWKV", layer_id=layer_id
        )

    def forward(self, x):
        return self.block(x)


# ==========================================================
# Straight-Through Top-K 自适应选择器
# ==========================================================
class StraightThroughScaleSelector(nn.Module):

    def __init__(self, time_steps, patch_len_list, hidden_dim=64, k_select=2, dropout=0.1):
        super().__init__()
        self.time_steps = time_steps
        self.patch_len_list = patch_len_list
        self.num_classes = len(patch_len_list)
        self.k_select = k_select
        self.scale_cls = nn.Sequential(
            nn.Linear(self.time_steps, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_classes)
        )

        # 用于 Budget Loss 的目标分布缓冲
        self.register_buffer('target_ratio', torch.ones(self.num_classes) / self.num_classes)
        self.latest_cls_soft = None

    def forward(self, x):

        logits = self.scale_cls(x)  # [N, num_classes]

        if self.training:

            u = torch.rand_like(logits)
            gumbel = -torch.log(-torch.log(u + 1e-10) + 1e-10)
            noisy_logits = logits + gumbel

            soft_probs = F.softmax(noisy_logits, dim=-1)

            _, top_k_indices = torch.topk(soft_probs, k=self.k_select, dim=-1)
            hard_mask = torch.zeros_like(soft_probs).scatter_(1, top_k_indices, 1.0)

            selection_weights = hard_mask + (soft_probs - soft_probs.detach())

            self.latest_cls_soft = F.softmax(logits, dim=-1)

        else:
            soft_probs = F.softmax(logits, dim=-1)
            _, top_k_indices = torch.topk(soft_probs, k=self.k_select, dim=-1)

            hard_mask = torch.zeros_like(logits).scatter_(1, top_k_indices, 1.0)
            selection_weights = hard_mask

            self.latest_cls_soft = soft_probs

        return selection_weights


def compute_budget_loss(adaptive_pool):
    device = adaptive_pool.target_ratio.device
    if adaptive_pool.latest_cls_soft is None:
        return torch.tensor(0.0, device=device)

    prob = adaptive_pool.latest_cls_soft.float().mean(dim=0)  # [num_classes]
    diff = prob - adaptive_pool.target_ratio
    budget_loss = torch.mean(diff ** 2)
    return budget_loss


# ==========================================================
# 结合稀疏计算的多尺度 RWKV 分支
# ==========================================================
class MultiScaleRWKVBranch(nn.Module):
    def __init__(self, time_steps, channels,
                 patch_len_list=(2, 4, 8, 16),
                 ie_hid=None, dropout=0.0, k_select=2):
        super().__init__()
        self.time_steps = time_steps
        self.channels = channels
        self.patch_len_list = patch_len_list
        self.k_select = min(k_select, len(patch_len_list))

        self.adaptive_selector = StraightThroughScaleSelector(
            time_steps=time_steps,
            patch_len_list=patch_len_list,
            hidden_dim=64,
            k_select=self.k_select,
            dropout=dropout
        )
        self.adaptive_pool = self.adaptive_selector

        self.rwkv_scales = nn.ModuleList()
        for L in self.patch_len_list:
            assert self.time_steps % L == 0, f"T={time_steps} not divisible by L={L}"
            T_scale = self.time_steps // L
            self.rwkv_scales.append(
                IEBlock(time_dim=T_scale, channel_dim=channels, hid_dim=ie_hid)
            )

        self.fusion_mlp = nn.Linear(channels, channels)
        self.rwkv_fine = IEBlock(time_dim=time_steps, channel_dim=channels, hid_dim=ie_hid)
        self.ar = nn.Linear(channels, channels)
        nn.init.xavier_uniform_(self.ar.weight)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        N, T, C = x.shape
        x_flat = x.mean(dim=2)  # [N, T]
        weights = self.adaptive_selector(x_flat)

        selection_mask = weights > 0.5

        adaptive_feat = torch.zeros_like(x)

        for i, (L, module) in enumerate(zip(self.patch_len_list, self.rwkv_scales)):

            mask_i = selection_mask[:, i]

            if not mask_i.any():
                continue

            x_sub = x[mask_i]  # [N_sub, T, C]

            curr_bs = x_sub.shape[0]
            x_input = x_sub.view(curr_bs, T // L, L, C).mean(dim=2)

            out = module(x_input)  # [N_sub, T/L, C]


            out_up = F.interpolate(
                out.transpose(1, 2), size=T, mode='linear', align_corners=False
            ).transpose(1, 2)


            w_sub = weights[mask_i, i].view(-1, 1, 1)

            weighted_out = out_up * w_sub


            adaptive_feat[mask_i] += weighted_out


        adaptive_feat = self.fusion_mlp(adaptive_feat)

        fine_feat = self.rwkv_fine(x)

        ar_feat = self.ar(x[:, -1, :]).unsqueeze(1).repeat(1, T, 1)

        final_out = fine_feat + adaptive_feat + ar_feat
        return self.dropout(final_out)


# ==========================================================
# NoGraphMixer
# ==========================================================
class NoGraphMixer(nn.Module):
    def __init__(self, stocks, hidden_dim=20):
        super(NoGraphMixer, self).__init__()
        self.dense1 = nn.Linear(stocks, hidden_dim)
        self.activation = nn.Hardswish()
        self.dense2 = nn.Linear(hidden_dim, stocks)
        self.layer_norm_stock = nn.LayerNorm(stocks)

    def forward(self, inputs):
        # inputs: [N, L]
        x = inputs  # [N, L]
        x = x.permute(1, 0)  # [L, N]
        x = self.layer_norm_stock(x)
        x = self.dense1(x)
        x = self.activation(x)
        x = self.dense2(x)
        x = x.permute(1, 0)  # [N, L]
        return x


# ==========================================================
# 主模型：StockMixer (集成改进后的多尺度分支)
# ==========================================================
class AMSFM(nn.Module):
    def __init__(self,
                 stocks,
                 time_steps, 
                 channels, 
                 market, 
                 scale, 
                 patch_len_list=(2, 4, 8, 16),
                 ie_hid=None,
                 dropout=0.0,
                 k_select=2):
        super(StockMixer, self).__init__()

        self.time_steps = time_steps
        self.channels = channels
        self.stocks = stocks

        self.lightts_branch = MultiScaleRWKVBranch(
            time_steps=time_steps,
            channels=channels,
            patch_len_list=patch_len_list,
            ie_hid=ie_hid,
            dropout=dropout,
            k_select=k_select  # 传递 k_select
        )

        self.channel_fc_a = nn.Linear(channels, 1)
        self.stock_mixer_a = NoGraphMixer(stocks=stocks, hidden_dim=market)
        self.time_fc_a = nn.Linear(time_steps, 1)
        self.time_fc_a_ = nn.Linear(time_steps, 1)

        self.channel_fc_b = nn.Linear(channels, 1)
        self.stock_mixer_b = NoGraphMixer(stocks=stocks, hidden_dim=market)
        self.time_fc_b = nn.Linear(time_steps, 1)
        self.time_fc_b_ = nn.Linear(time_steps, 1)

        self.gate_param = nn.Parameter(torch.tensor(-1.0))

    def branch_forward(self, feat, channel_fc, stock_mixer, time_fc, time_fc_):
        # feat: [N, T, C]
        y_ch = channel_fc(feat).squeeze(-1)  # [N, T]
        z = stock_mixer(y_ch)  # [N, T]
        y_time = time_fc(y_ch)  # [N, 1]
        z_time = time_fc_(z)  # [N, 1]
        return y_time + z_time  # [N, 1]

    def forward(self, inputs):
        """
        inputs: [N, T, C]
        return: [N, 1]
        """

        feat = self.lightts_branch(inputs)  # [N, T, C]


        pred_a = self.branch_forward(
            feat=feat,
            channel_fc=self.channel_fc_a,
            stock_mixer=self.stock_mixer_a,
            time_fc=self.time_fc_a,
            time_fc_=self.time_fc_a_,
        )


        pred_b = self.branch_forward(
            feat=feat,
            channel_fc=self.channel_fc_b,
            stock_mixer=self.stock_mixer_b,
            time_fc=self.time_fc_b,
            time_fc_=self.time_fc_b_,
        )


        gate = torch.sigmoid(self.gate_param).view(1, 1)
        pred = (1.0 - gate) * pred_a + gate * pred_b
        return pred
