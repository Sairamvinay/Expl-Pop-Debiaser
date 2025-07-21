import torch
import torch.nn as nn
import torch.nn.functional as F


# Stage 2: Fairness Loss
def fairness_disparity_loss(pop_scores, niche_scores, disp_ratio):
    pop_sum = pop_scores.sum()
    niche_sum = niche_scores.sum()
    fairness_loss = ((pop_sum - disp_ratio * niche_sum) / (pop_sum + niche_sum + 1e-8)).pow(2)
    return fairness_loss


def utility_relevance_loss(labels, scores):
    relevance_loss = F.binary_cross_entropy_with_logits(scores, labels)
    return relevance_loss

# Stage 1: BPR Explanation Pairwise Loss
def bpr_loss(pos_score, neg_score):
    diff = pos_score - neg_score
    diff = diff.clamp(min=-10, max=10)
    return -torch.mean(F.logsigmoid(diff))

# Zero explanation embedding for neutral scoring
def get_zero_explanation_embedding(model, tokenizer, device, batch_size, max_length=512):
    input_ids = torch.full((batch_size, max_length), tokenizer.pad_token_id, dtype=torch.long).to(device)
    attention_mask = torch.ones((batch_size, max_length), dtype=torch.long).to(device)  # force decoder to ignore all tokens

    with torch.no_grad():
        output = model(input_ids=input_ids, attention_mask=attention_mask)

    # Safe zero vector if required
    zero_embed = output.hidden_states[-1].mean(dim=1)
    
    # Optional override to forcefully use pure zero vector
#     if torch.any(zero_embed != 0):
#         zero_embed = torch.zeros_like(zero_embed)

    return zero_embed  # (1, D)

# Helper to get explanation embeddings from frozen LLaMA model
def get_explanation_embedding(encoder, tokenizer, text, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True,max_length=512).to(device)
    with torch.no_grad():
        output = encoder(**inputs)
    
    return output.hidden_states[-1].mean(dim=1)  # Mean pooling

class DeepFM(nn.Module):
    def __init__(self, user_num, item_num, id_embed_dim, expl_embed_dim, hidden_dims, dropout_rate = 0.2):
        super().__init__()
        self.user_embed = nn.Embedding(user_num, id_embed_dim)
        self.item_embed = nn.Embedding(item_num, id_embed_dim)
        
        expl_dim = id_embed_dim

        # Linear (first-order term)
        self.linear = nn.Linear(id_embed_dim * 2 + expl_embed_dim, 1)
        
        # self.linear_norm = nn.LayerNorm(1) # ADDED JUNE 18
        # self.fm_norm = nn.LayerNorm(1) # ADDED JUNE 18
        
        # self.final_norm = nn.LayerNorm(1)
        
        
        self.input_norm = nn.LayerNorm(id_embed_dim * 2 + expl_embed_dim)

        self.expl_proj = nn.Linear(expl_embed_dim, expl_dim)

        # Deep component
        deep_input_dim = id_embed_dim * 2 + expl_embed_dim
        layers = []
        for h in hidden_dims:
            layers.append(nn.Linear(deep_input_dim, h))
            layers.append(nn.LayerNorm(h)) # ADDED JUNE 18
            layers.append(nn.ReLU())
            deep_input_dim = h
        self.deep = nn.Sequential(*layers)
        self.deep_out = nn.Linear(hidden_dims[-1], 1)
        
        nn.init.xavier_uniform_(self.linear.weight, gain=0.1)
        nn.init.xavier_uniform_(self.deep_out.weight, gain=0.1)
        nn.init.xavier_uniform_(self.expl_proj.weight, gain=0.1)

        # FM second-order interaction
        self.dropout = nn.Dropout(dropout_rate)
    
    def fm_interaction(self, x):
        sum_square = torch.sum(x, dim=1) ** 2
        square_sum = torch.sum(x ** 2, dim=1)
        interaction = 0.5 * (sum_square - square_sum)
        return torch.sum(interaction, dim=1, keepdim=True)

    def forward(self, user_ids, item_ids, expl_embeds,mild_factor_scale=0.4):
        u = self.user_embed(user_ids)
        i = self.item_embed(item_ids)
        
        expl_embeds = F.normalize(expl_embeds, p=2, dim=-1)
        
        e = self.expl_proj(expl_embeds)

        feats = torch.stack([u, i, e], dim=1)  # shape (B, 3, D)
        concat_feats = torch.cat([u, i, expl_embeds], dim=-1)  # shape (B, 2*ID_dim + expl_dim)
        # concat_feats = torch.cat([u, i, e],dim=-1) # shape (B, 3* ID_dim)
        concat_feats = self.input_norm(concat_feats)


        linear_out = self.linear(concat_feats) # self.linear_norm(self.linear(concat_feats))
        fm_out = self.fm_interaction(feats) # self.fm_norm(self.fm_interaction(feats))
        
        deep_out = self.deep(concat_feats)
        deep_out = self.deep_out(self.dropout(deep_out))
        
        score = linear_out + fm_out + deep_out
        score *= mild_factor_scale
        # score = torch.clamp(score, min=-10.0, max=10.0)  # prevent exploding logits
        return score

    def freeze_component(self, component_names):
        for name in component_names:
            module = getattr(self, name, None)
            if module is not None:
                for param in module.parameters():
                    param.requires_grad = False
                print(f"param: {name} is frozen")
    