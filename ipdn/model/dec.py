import torch
import torch.nn as nn
from .sample_model import SamplingModule
import torch.nn.functional as F
from ..torch.nn import MultiheadAttention

class ThreeLayerMLP(nn.Module):
    """A 3-layer MLP with normalization and dropout."""

    def __init__(self, dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dim, dim, 1, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(dim, dim, 1, bias=False),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(dim, out_dim, 1)
        )

    def forward(self, x):
        """Forward pass, x can be (B, dim, N)."""
        return self.net(x)

class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model=256, nhead=8, dropout=0.0):
        super().__init__()
        self.attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, source, query, batch_mask=None, attn_mask=None, pe=None):
        """
        source (B, N_p, d_model)
        batch_offsets Tensor (b, n_p)
        query Tensor (b, n_q, d_model)
        attn_masks Tensor (b, n_q, n_p)
        """
        B = query.shape[0]
        query = self.with_pos_embed(query, pe)
        k = v = source
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1).view(B*self.nhead, query.shape[1], k.shape[1])
            output, output_weight, src_weight = self.attn(query, k, v, key_padding_mask=batch_mask, attn_mask=attn_mask)  # (1, 100, d_model)
        else:
            output, output_weight, src_weight = self.attn(query, k, v, key_padding_mask=batch_mask)
        self.dropout(output)
        output = output + query
        self.norm(output)

        return output, output_weight, src_weight # (b, n_q, d_model), (b, n_q, n_v)

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model=256, nhead=8, dropout=0.0, glu = False):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.nhead = nhead
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.glu = glu
        if glu:
            self.glu_p = nn.Linear(d_model, d_model)
            self.glu_g = nn.Linear(d_model, d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, x, x_mask=None, attn_mask=None, pe=None, norm=True):
        """
        x Tensor (b, n_w, c)
        x_mask Tensor (b, n_w)
        """
        B = x.shape[0]
        q = k = self.with_pos_embed(x, pe)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.nhead, 1, 1).view(B*self.nhead, q.shape[1], k.shape[1])
            output, _ = self.attn(q, k, x, key_padding_mask=x_mask, attn_mask=attn_mask)  # (1, 100, d_model)
        else:
            output, _ = self.attn(q, k, x, key_padding_mask=x_mask)
        if self.glu:
            output = self.glu_p(output) * (self.glu_g(output).sigmoid())
        output = self.dropout(output) + x
        if norm: output = self.norm(output)
        return output

class FFN(nn.Module):

    def __init__(self, d_model, hidden_dim, dropout=0.0, activation_fn='relu'):
        super().__init__()
        if activation_fn == 'relu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        elif activation_fn == 'gelu':
            self.net = nn.Sequential(
                nn.Linear(d_model, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, d_model),
                nn.Dropout(dropout),
            )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, norm=True):
        output = self.net(x)
        output = output + x
        if norm: output = self.norm(output)
        return output
    
class DEC(nn.Module):
    """
    in_channels List[int] (4,) [64,96,128,160]
    """
    def __init__(
        self,
        num_layer=6,
        num_class=256,
        in_channel=32,
        d_model=256,
        nhead=8,
        hidden_dim=1024,
        dropout=0.0,
        activation_fn='relu',
        iter_pred=False,
        attn_mask=False,
        sampling_module=None,
        kernel='top1',
        global_feat='mean',
        lang_att=False,
        contrastive_align_loss=False,
    ):
        super().__init__()
        self.num_layer = num_layer
        self.num_class = num_class
        self.d_model = d_model
        self.input_proj = nn.Sequential(nn.Linear(in_channel, d_model), nn.LayerNorm(d_model), nn.ReLU())
        
        self.input_proj_2d = nn.Sequential(nn.Linear(1024, d_model*2),nn.ReLU(),nn.Linear(d_model*2, d_model),nn.ReLU(),nn.Linear(d_model, d_model))#clip
        self.sum_norm = nn.LayerNorm(d_model)
        
        self.lang_att = lang_att
        self.contrastive_align_loss = contrastive_align_loss

        H = 768
        self.lang_proj = nn.Linear(H, d_model)
        self.lang_norm = nn.LayerNorm(d_model)
        
        if sampling_module is not None:
            self.sampling_module = SamplingModule(**sampling_module)
        else:
            self.sampling_module = None

        self.query_generator = nn.Sequential(nn.Linear(d_model, d_model),nn.ReLU(),nn.Linear(d_model, d_model),nn.ReLU(),nn.Linear(d_model, d_model))
        
        # DDI and SWA
        self.swa_layers = nn.ModuleList([])
        self.rra_layers = nn.ModuleList([])
        self.rla_layers = nn.ModuleList([])
        self.swa_ffn_layers = nn.ModuleList([])
        self.sem_cls_heads = nn.ModuleList([])
        self.scg = nn.ModuleList([])
        self.lqg = nn.ModuleList([])
        if self.lang_att:
            self.lla_layers = nn.ModuleList([])
            self.lsa_layers = nn.ModuleList([])
            self.lsa_ffn_layers = nn.ModuleList([])
        for i in range(num_layer):
            self.swa_layers.append(CrossAttentionLayer(d_model, nhead, dropout))
            self.rra_layers.append(SelfAttentionLayer(d_model, nhead, dropout))
            self.rla_layers.append(CrossAttentionLayer(d_model, nhead, dropout))
            self.swa_ffn_layers.append(FFN(d_model, hidden_dim, dropout, activation_fn))
            self.sem_cls_heads.append(ThreeLayerMLP(d_model, self.num_class))
            self.scg.append(SelfAttentionLayer(d_model, nhead, dropout))
            self.lqg.append(CrossAttentionLayer(d_model, nhead, dropout))
            if self.lang_att:
                self.lla_layers.append(SelfAttentionLayer(d_model, nhead, dropout))
                self.lsa_layers.append(CrossAttentionLayer(d_model, nhead, dropout))
                self.lsa_ffn_layers.append(FFN(d_model, hidden_dim, dropout, activation_fn))    
        
        self.sem_cls_head = ThreeLayerMLP(d_model, self.num_class)

        self.out_norm = nn.LayerNorm(d_model)
        self.out_score = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 1))
        self.x_mask = nn.Sequential(nn.Linear(in_channel, d_model), nn.ReLU(), nn.Linear(d_model, d_model))
        
        self.indi_embedding = nn.Sequential(nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 2), nn.Linear(2, 2))
        self.indi_norm = nn.LayerNorm(d_model)

        self.iter_pred = iter_pred
        self.attn_mask = attn_mask
        self.kernel = kernel
        self.global_feat = global_feat
        
        self.k = 8

        # Extra layers for contrastive losses
        if contrastive_align_loss:
            self.contrastive_align_projection_vision = nn.Sequential(nn.Linear(d_model, d_model),nn.ReLU(),nn.Linear(d_model, d_model),nn.ReLU(),nn.Linear(d_model, 64))
            self.contrastive_align_projection_text = nn.Sequential(nn.Linear(d_model, d_model),nn.ReLU(),nn.Linear(d_model, d_model),nn.ReLU(),nn.Linear(d_model, 64))         
        
        self.scg_head = SelfAttentionLayer(d_model, nhead, dropout=0, glu=True)
        
    
    def forward_iter_pred(self, x, fps_seed_sp, batch_offsets, lang_feats=None, lang_masks=None, sp_pos=None, feats_2d=None):
        """
        x [B*M, inchannel]
        """
        
        lang_feats = self.lang_proj(lang_feats) 
        lang_feats = self.lang_norm(lang_feats)  
        lang_masks = ~(lang_masks.bool())
        inst_feats = self.input_proj(x)
        
        inst_feats_2d = self.input_proj_2d(feats_2d)
        inst_feats = self.sum_norm(inst_feats+inst_feats_2d)
        
        mask_feats = self.x_mask(x)
        inst_feats, batch_mask = self.get_batches(inst_feats, batch_offsets)
        mask_feats, _ = self.get_batches(mask_feats, batch_offsets)
        
        sp_pos, _ = self.get_batches(sp_pos, batch_offsets)
        
        prediction_masks, prediction_scores, prediction_classes, prediction_indis = [], [], [], []
        
        sample_inds = None
        ref_scores = None
         
        seed_sp = inst_feats.gather(dim=1, index=fps_seed_sp.long().unsqueeze(-1).repeat(1, 1, inst_feats.size(-1)))
        seed_sp_pos = sp_pos.gather(dim=1, index=fps_seed_sp.long().unsqueeze(-1).repeat(1, 1, sp_pos.size(-1)))
        
        scg_mask = self.get_scg_mask(seed_sp_pos)
        seed_sp = self.scg_head(seed_sp, attn_mask = ~scg_mask)

         # sampling
        if hasattr(self, 'sampling_module') and self.sampling_module is not None:
            sample_inds, ref_scores = self.sampling_module(seed_sp, lang_feats, None, lang_masks)
            sample_inds = sample_inds.long()
            sampled_seed = seed_sp.gather(dim=1, index=sample_inds.unsqueeze(-1).repeat(1, 1, seed_sp.size(-1)))
            query_pos = seed_sp_pos.gather(dim=1, index=sample_inds.unsqueeze(-1).repeat(1, 1, seed_sp_pos.size(-1)))
            query = self.query_generator(sampled_seed)
        else:
            query = self.query_generator(seed_sp)
            query_pos = seed_sp_pos
            
        scg_mask = self.get_scg_mask(query_pos)

        proj_queries = []
        if self.contrastive_align_loss:
            proj_queries.append(F.normalize(self.contrastive_align_projection_vision(query), p=2, dim=-1))
        else:
            proj_queries.append(None)
            
        proj_tokens = []
        if self.contrastive_align_loss:
            proj_tokens.append(F.normalize(self.contrastive_align_projection_text(lang_feats), p=2, dim=-1))
        else:
            proj_tokens.append(None)
        
        
        pred_scores, pred_masks, attn_masks = self.prediction_head(query, mask_feats, batch_mask)
        pred_indis = self.indi_embedding(query)
        prediction_scores.append(pred_scores)
        prediction_masks.append(pred_masks)
        prediction_indis.append(pred_indis)
        
        init_lang_mask = lang_masks
        #lang_query = lang_feats
        
        # multi-round
        l = 0
        for i in range(self.num_layer):
            #PAD
            lang_query, w1, w2 = self.lqg[i](query[:,:128],lang_feats)
            if i > l:
                mask = pred_indis.softmax(-1)[:,:,1]>0.75
                weight = (w2 * (~init_lang_mask).unsqueeze(-1)).sum(1)
                weight = torch.masked_fill(weight,~mask,float('-inf')).softmax(-1)
                weight = torch.masked_fill(weight, torch.isnan(weight), 0)
                #prompt = torch.einsum('abc,acd->abd',weight.unsqueeze(1),query[:,:128])
                prompt = weight.unsqueeze(-1) * (query[:,:128])
                #prompt = self.prompt[i].unsqueeze(0).repeat(query.shape[0],1,1)
                query = torch.cat([query[:,:128], prompt], dim=1)
                _, _, attn_masks = self.prediction_head(query, mask_feats, batch_mask)
            
            
            if self.lang_att:
                lang_query = self.lla_layers[i](lang_query, lang_masks)
                lang_query, _, _ = self.lsa_layers[i](inst_feats, lang_query, batch_mask, None)
                lang_query = self.lsa_ffn_layers[i](lang_query)

            query, _, _ = self.swa_layers[i](inst_feats, query, batch_mask, attn_masks)
            query_rra = self.rra_layers[i](query)
            query_rla, _, _ = self.rla_layers[i](lang_query, query, lang_masks)
            
            if self.lang_att:
                lang_query = self.lla_layers[i](lang_query, lang_masks)
                lang_query, _, _ = self.lsa_layers[i](query, lang_query)
                lang_query = self.lsa_ffn_layers[i](lang_query)

            query = query + query_rla + query_rra 
            
            if i <= l:
                query = self.scg[i](query, attn_mask=~scg_mask)
            else:
                query = torch.cat([self.scg[i](query[:,:128], attn_mask=~scg_mask),query[:,128:]],dim=1)
            
            query = self.swa_ffn_layers[i](query)


            pred_scores, pred_masks, attn_masks = self.prediction_head(query[:,:128], mask_feats, batch_mask)
            pred_indis = self.indi_embedding(query[:,:128])
            prediction_scores.append(pred_scores)
            prediction_masks.append(pred_masks)
            prediction_indis.append(pred_indis)
            if self.contrastive_align_loss:
                proj_queries.append(F.normalize(self.contrastive_align_projection_vision(query[:,:128]), p=2, dim=-1))
            else:
                proj_queries.append(None)
                
            if self.contrastive_align_loss:
                proj_tokens.append(F.normalize(self.contrastive_align_projection_text(lang_query), p=2, dim=-1))
            else:
                proj_tokens.append(None)

        return {
            'masks': pred_masks,
            'batch_mask': batch_mask,
            'scores': pred_scores,
            'indis': pred_indis, # [B, B_q, 2]
            'proj_queries': proj_queries[-1],
            'proj_tokens': proj_tokens[-1],
            'sample_inds': sample_inds, # [B, K]
            'ref_scores': ref_scores, # [B, M]
            'aux_outputs': [{
                'masks': a,
                'scores': b,
                'proj_queries': c,
                'indis': d,
                'proj_tokens': e,
            } for a, b, c, d, e in zip(
                prediction_masks[:-1],
                prediction_scores[:-1],
                proj_queries[:-1],
                prediction_indis[:-1],
                proj_tokens[:-1],
            )],
        }
        
    def get_batches(self, x, batch_offsets):
        B = len(batch_offsets) - 1
        max_len = max(batch_offsets[1:] - batch_offsets[:-1])
        if torch.is_tensor(max_len):
            max_len = max_len.item()
        new_feats = torch.zeros(B, max_len, x.shape[1]).to(x.device)
        mask = torch.ones(B, max_len, dtype=torch.bool).to(x.device)
        for i in range(B):
            start_idx = batch_offsets[i]
            end_idx = batch_offsets[i + 1]
            cur_len = end_idx - start_idx
            padded_feats = torch.cat([x[start_idx:end_idx], torch.zeros(max_len - cur_len, x.shape[1]).to(x.device)], dim=0)
            new_feats[i] = padded_feats
            mask[i, :cur_len] = False
        mask.detach()
        return new_feats, mask
    
    def get_mask(self, query, mask_feats, batch_mask):
        pred_masks = torch.einsum('bnd,bmd->bnm', query, mask_feats)
        if self.attn_mask:
            attn_masks = (pred_masks.sigmoid() < 0.5).bool() # [B, 1, num_sp]
            attn_masks = attn_masks | batch_mask.unsqueeze(1)
            attn_masks[torch.where(attn_masks.sum(-1) == attn_masks.shape[-1])] = False
            attn_masks = attn_masks | batch_mask.unsqueeze(1)
            attn_masks = attn_masks.detach()
        else:
            attn_masks = None
        return pred_masks, attn_masks

    def prediction_head(self, query, mask_feats, batch_mask):
        query = self.out_norm(query)
        pred_scores = self.out_score(query)
        pred_masks, attn_masks = self.get_mask(query, mask_feats, batch_mask)
        return pred_scores, pred_masks, attn_masks

    def get_scg_mask(self, pos_q, pos_k=None, k_mask=None, k=8):
        if pos_k == None: pos_k = pos_q
        scg_mask = torch.zeros((pos_q.shape[0],pos_q.shape[1],pos_k.shape[1]),device=pos_q.device).bool()
        dis = ((pos_q.unsqueeze(2)-pos_k.unsqueeze(1))**2).sum(-1)
        if k_mask is not None:
            dis = torch.masked_fill(dis, k_mask.unsqueeze(1).repeat(1,pos_q.shape[1],1), 1000000.0)
        ind = torch.topk(dis, k, dim=-1, largest=False)[1]
        for i in range(ind.shape[0]):
            for j in range(ind.shape[1]):
                scg_mask[i,j,ind[i][j]] = True
        return scg_mask
    
    def forward(self, x, fps_seed_sp, batch_offsets, lang_feats=None, lang_masks=None, sp_pos=None, feats_2d=None):
        if self.iter_pred:
            return self.forward_iter_pred(x, fps_seed_sp, batch_offsets, lang_feats, lang_masks, sp_pos, feats_2d)
        else:
            raise NotImplementedError
