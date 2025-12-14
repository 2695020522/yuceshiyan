import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class GaussianPriorAttention(nn.Module):
    def __init__(self, d_model, num_heads, scale_level=0, patch_time_span=1, enable_prior=True):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.enable_prior = enable_prior
        
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # 优化：根据尺度调整 Sigma，粗尺度 Sigma 应该更小（因为1个点代表多天）
        effective_stride = (2 ** scale_level) * patch_time_span
        self.periods = [1.0 / effective_stride, 30.0 / effective_stride] 
        self.weights = [0.8, 0.4] # 调大权重
        self.sigmas = [0.5, 2.0]  
        
        # 关键手段：初始化 Lambda 为 2.0，强迫模型初期关注先验
        self.prior_lambda = nn.Parameter(torch.tensor(2.0))

    def _generate_gaussian_mask(self, length, device):
        indices = torch.arange(length, device=device).float()
        i = indices.view(-1, 1)
        j = indices.view(1, -1)
        delta = torch.abs(i - j)
        
        mask = torch.zeros_like(delta)
        for p, w, sigma in zip(self.periods, self.weights, self.sigmas):
            if p < 0.2: continue 
            term = torch.exp(-((delta - p)**2) / (2 * sigma**2))
            mask += w * term
        return mask

    def forward(self, x):
        B, L, D = x.shape
        Q = self.query(x).view(B, L, self.num_heads, D // self.num_heads).transpose(1, 2)
        K = self.key(x).view(B, L, self.num_heads, D // self.num_heads).transpose(1, 2)
        V = self.value(x).view(B, L, self.num_heads, D // self.num_heads).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(D // self.num_heads)
        
        if self.enable_prior:
            prior_mask = self._generate_gaussian_mask(L, x.device)
            scores = scores + self.prior_lambda * prior_mask.unsqueeze(0).unsqueeze(0)
        
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out)

class AdaptiveGatedBlock(nn.Module):
    def __init__(self, d_model, enable_gating=True):
        super().__init__()
        self.enable_gating = enable_gating
        
        self.time_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU())
        self.val_proj = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU())
        
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
        self.cross_trans = nn.Linear(d_model, d_model)

    def forward(self, x):
        h_time = self.time_proj(x)
        h_val = self.val_proj(x)
        
        if not self.enable_gating:
            return h_time, h_val
        
        combined = torch.cat([h_time, h_val], dim=-1)
        gate = self.gate_net(combined)
        h_val_fused = h_val + gate * self.cross_trans(h_time)
        return h_time, h_val_fused

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, stride, in_dim, d_model):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        self.proj = nn.Linear(in_dim * patch_size, d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, 100, d_model)) 

    def forward(self, x):
        B, L, D = x.shape
        patches = x.unfold(1, self.patch_size, self.stride) 
        patches = patches.permute(0, 1, 3, 2).reshape(B, -1, D * self.patch_size)
        emb = self.proj(patches)
        emb = emb + self.pos_embed[:, :emb.size(1), :]
        return emb

class ScaleEncoder(nn.Module):
    def __init__(self, scale_level, d_model, enable_prior, enable_gating):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_size=4, stride=2, in_dim=4, d_model=d_model)
        self.attn = GaussianPriorAttention(d_model, num_heads=4, scale_level=scale_level, enable_prior=enable_prior)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_model*4), nn.ReLU(), nn.Linear(d_model*4, d_model))
        self.norm2 = nn.LayerNorm(d_model)
        self.gating = AdaptiveGatedBlock(d_model, enable_gating=enable_gating)

    def forward(self, x):
        h = self.patch_embed(x)
        attn_out = self.attn(h)
        h = self.norm1(h + attn_out)
        ffn_out = self.ffn(h)
        h_enc = self.norm2(h + ffn_out)
        h_time, h_val = self.gating(h_enc)
        return h_time, h_val

class MultiScaleTransformer(nn.Module):
    def __init__(self, d_model=64, enable_prior=True, enable_gating=True):
        super().__init__()
        self.scales = 3
        self.downsample = nn.AvgPool1d(kernel_size=2, stride=2)
        self.encoders = nn.ModuleList([ScaleEncoder(k, d_model, enable_prior, enable_gating) for k in range(self.scales)])
        self.time_fusion_attn = nn.Linear(d_model, 1)
        self.val_fusion_attn = nn.Linear(d_model, 1)
        self.head_time = nn.Linear(d_model, 1)
        self.head_val = nn.Linear(d_model, 2)

    def forward(self, x):
        inputs = [x]
        curr = x
        for _ in range(self.scales - 1):
            curr = curr.permute(0, 2, 1)
            curr = self.downsample(curr)
            curr = curr.permute(0, 2, 1)
            inputs.append(curr)
            
        time_feats, val_feats = [], []
        for k in range(self.scales):
            h_t, h_v = self.encoders[k](inputs[k])
            time_feats.append(h_t.mean(dim=1)) 
            val_feats.append(h_v.mean(dim=1))
            
        stack_time = torch.stack(time_feats, dim=1)
        stack_val = torch.stack(val_feats, dim=1)
        
        scores_t = F.softmax(self.time_fusion_attn(stack_time), dim=1)
        V_time = (stack_time * scores_t).sum(dim=1)
        
        scores_v = F.softmax(self.val_fusion_attn(stack_val), dim=1)
        V_val = (stack_val * scores_v).sum(dim=1)
        
        pred_time = self.head_time(V_time)
        pred_val = self.head_val(V_val)
        return pred_time, pred_val