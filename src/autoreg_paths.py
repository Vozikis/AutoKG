
#this has extra random permutation on the training set.


import os
import math, torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from intelligraphs.data_loaders import load_data_as_list
from tqdm import tqdm
import random
from intelligraphs.evaluators import post_process_data, SemanticEvaluator
from intelligraphs.verifier.synthetic import SynPathsVerifier   

CHECKPOINT = "syn_paths.pt"  
RESUME     = True        

def save_ckpt(epoch):
    torch.save(
        {
            "epoch"  : epoch,                   
            "model"  : model.state_dict(),      
            "opt"    : opt.state_dict(),        
            "sched"  : sched.state_dict(),      
        },
        CHECKPOINT)

def load_ckpt():
    ckpt = torch.load(CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    sched.load_state_dict(ckpt["sched"])
    return ckpt["epoch"] + 1                   





total_epochs   = 300
batch_size     = 256
d_model        = 512
latent_dim     = 10
nhead          = 4
num_layers     = 3
beam_width     = 4          
eval_every     = 400



(train_g, val_g, test_g,
 (e2i, i2e), (r2i, i2r), *_ ) = load_data_as_list("syn-paths")

num_entities   = len(e2i)                                   # 49
num_relations  = len(r2i)                                   #  3
max_triples    = max(len(g) for g in train_g+val_g+test_g)   # 3
SPECIAL        = {"PAD": 0, "BOS": 1, "EOS": 2}
ENT_BASE       = 3
REL_BASE       = ENT_BASE + num_entities
VOCAB_SIZE     = REL_BASE + num_relations
seq_len        = 1 + max_triples*3 + 1                      # 11


def canonicalize(triples, mode="alpha_name"):
    if mode == "keep":
        return triples

    if mode == "alpha_name":
        return sorted(triples,
                      key=lambda x: (i2e[x[0]], i2r[x[1]], i2e[x[2]]))

    return sorted(triples,
                      key=lambda x: (i2e[x[0]], i2r[x[1]], i2e[x[2]]))

def triples_to_seq(triples):
    seq = [SPECIAL["BOS"]]
    for h, r, t in triples:
        seq += [ENT_BASE+h, REL_BASE+r, ENT_BASE+t]
    seq.append(SPECIAL["EOS"])
    seq += [SPECIAL["PAD"]] * (seq_len - len(seq))
    return torch.tensor(seq, dtype=torch.long)

def seq_to_triples(seq):
    triples, i = [], 1
    while i+2 < len(seq) and seq[i] != SPECIAL["EOS"]:
        h, r, t = seq[i:i+3].tolist()
        triples.append((h-ENT_BASE, r-REL_BASE, t-ENT_BASE))
        i += 3
    return triples                      

class GraphSeqDataset(Dataset):
    def __init__(self, graphs,
                 triple_order="alpha_name",      
                 permute=False):
        self.graphs = [canonicalize(g, triple_order) for g in graphs]

        self.permute = permute

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        triples = self.graphs[idx]
        if self.permute:                          
            triples = random.sample(triples, k=len(triples))

        triples_tensor = torch.tensor(triples, dtype=torch.long)
        seq_tensor     = triples_to_seq(triples)
        return triples_tensor, seq_tensor


class GraphEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.e_emb = nn.Embedding(num_entities,  d_model)
        self.r_emb = nn.Embedding(num_relations, d_model)
        self.txf   = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model*3, nhead, batch_first=True), 2)
        self.mu    = nn.Linear(d_model*3, latent_dim)
        self.logv  = nn.Linear(d_model*3, latent_dim)
    def forward(self, triples):
        h = self.e_emb(triples[:,:,0])
        r = self.r_emb(triples[:,:,1])
        t = self.e_emb(triples[:,:,2])
        x = self.txf(torch.cat([h, r, t], -1)).mean(1)
        mu, logv = self.mu(x), self.logv(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logv)
        return z, mu, logv


class AutoRegDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_emb = nn.Embedding(seq_len,   d_model)
        self.z_proj  = nn.Linear(latent_dim,   d_model)
        layer  = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
        self.txf = nn.TransformerDecoder(layer, num_layers)
        self.out = nn.Linear(d_model, VOCAB_SIZE)
    def forward(self, z, tgt):                      
        B, L  = tgt.shape
        tok   = self.tok_emb(tgt)
        pos   = self.pos_emb(torch.arange(L, device=z.device)).unsqueeze(0)
        mem   = self.z_proj(z).unsqueeze(1).repeat(1, L, 1)
        mask  = torch.triu(torch.ones(L, L, device=z.device, dtype=torch.bool), 1)
        return self.out(self.txf(tok+pos, mem, tgt_mask=mask))


class Model(nn.Module):
    def __init__(self): super().__init__(); self.enc, self.dec = GraphEncoder(), AutoRegDecoder()
    def forward(self, triples, seq_in):             
        z, mu, logv = self.enc(triples)
        return self.dec(z, seq_in), mu, logv

@torch.no_grad()
def generate(model, triples, beam=beam_width):
    device = next(model.parameters()).device
    B      = triples.size(0)
    z, *_  = model.enc(triples.to(device))
    BOS    = torch.full((B, 1), SPECIAL["BOS"], device=device)
    seqs   = [(BOS, torch.zeros(B, device=device))]           
    for _ in range(seq_len-1):
        cand = []
        for s, lp in seqs:
            logp = F.log_softmax(model.dec(z, s)[:, -1], -1)
            top_lp, top_id = logp.topk(beam, -1)
            for k in range(beam):
                cand.append((torch.cat([s, top_id[:, k, None]], 1), lp + top_lp[:, k]))
        seqs = sorted(cand, key=lambda x: x[1].mean().item(), reverse=True)[:beam]
        if all((s[:, -1] == SPECIAL["EOS"]).all() for s, _ in seqs):
            break
    best = seqs[0][0].cpu()
    return [seq_to_triples(s) for s in best]

# def strict_graph_accuracy(pred, gt):
#     return 100.0 * sum(p == g for p, g in zip(pred, gt)) / len(gt)

def kl(mu, logv): return -0.5 * torch.mean(1 + logv - mu.pow(2) - logv.exp())



train_loader = DataLoader(
    GraphSeqDataset(train_g,
                    triple_order="alpha_name",   
                    permute=False),           
    batch_size=batch_size,
    shuffle=True,          
    drop_last=True)

val_loader  = DataLoader(
    GraphSeqDataset(val_g,  triple_order="alpha_name", permute=False),
    batch_size)

test_loader = DataLoader(
    GraphSeqDataset(test_g, triple_order="alpha_name", permute=False),
    batch_size)



# def ints_to_labels(graph):
#     """(h,r,t) id-triples  →  human-readable labels."""
#     return [(i2e[h], i2r[r], i2e[t]) for h, r, t in graph]

# print("\n=== Sanity-check: first 10 training graphs ===")
# ds_check = GraphSeqDataset(train_g,
#                            triple_order="alpha_name",   # or "alpha_name"
#                            permute=False)

# for i in range(10):
#     triples_int, _ = ds_check[i]          # (tensor, seq) → take the triples
#     triples_int = triples_int.tolist()
#     print(f"\nGraph #{i:02d} (IDs):      {triples_int}")
#     print(  f"Graph #{i:02d} (labels):   {ints_to_labels(triples_int)}")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = Model().to(device)
opt    = torch.optim.Adam(model.parameters(), 1e-4)
sched  = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=total_epochs, eta_min=1e-6)

beta0, beta1 = 0.1, 1.0
total_losses, ce_losses, kl_losses, strict_acc = [], [], [], []

start_epoch = load_ckpt() if (RESUME and os.path.exists(CHECKPOINT)) else 0
for epoch in range(start_epoch, total_epochs):
    model.train()
    train_loader = DataLoader(
    GraphSeqDataset(train_g, triple_order="alpha_name", permute=False),   
    batch_size=batch_size, shuffle=True, drop_last=True
)

    beta = beta0 + (beta1 - beta0) * epoch / total_epochs
    ce_sum = kl_sum = tot_sum = 0.0
    batches = 0

    for triples, seq in train_loader:
        triples, seq = triples.to(device), seq.to(device)
        logits, mu, logv = model(triples, seq[:, :-1])
        ce   = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE),
                               seq[:, 1:].reshape(-1),
                               ignore_index=SPECIAL["PAD"])
        kl_v = kl(mu, logv)
        loss = ce + beta * kl_v
        opt.zero_grad(); loss.backward(); opt.step()
        ce_sum += ce.item(); kl_sum += kl_v.item(); tot_sum += loss.item()
        batches += 1
    sched.step()

    total_epoch = tot_sum / batches
    ce_epoch    = ce_sum / batches
    kl_epoch    = kl_sum / batches
    print(f"Epoch {epoch+1:3d}/{total_epochs} | "
          f"Total {total_epoch:8.4f} | CE {ce_epoch:8.4f} | KL {kl_epoch:8.4f}")

    total_losses.append(total_epoch)
    ce_losses.append(ce_epoch)
    kl_losses.append(kl_epoch)

    save_ckpt(epoch)







def ints_to_labels(graphs):
    return [[(i2e[h], i2r[r], i2e[t]) for h, r, t in g] for g in graphs]


@torch.no_grad()
def decode_latent(model, z, beam=beam_width):
    z   = z.to(next(model.parameters()).device, dtype=torch.float32)
    B   = z.size(0)
    BOS = torch.full((B, 1), SPECIAL["BOS"], dtype=torch.long, device=z.device)
    seqs = [(BOS, torch.zeros(B, device=z.device))]

    for _ in range(seq_len - 1):
        cand = []
        for s, lp in seqs:
            logits      = model.dec(z, s)[:, -1]
            logp        = F.log_softmax(logits, dim=-1)
            top_lp, ids = logp.topk(beam, dim=-1)          
            for k in range(beam):
                cand.append((torch.cat([s, ids[:, k, None]], 1),
                             lp + top_lp[:, k]))
        seqs = sorted(cand, key=lambda x: x[1].mean().item(),
                      reverse=True)[:beam]
        if all((s[:, -1] == SPECIAL["EOS"]).all() for s, _ in seqs):
            break

    best = seqs[0][0].cpu()
    return [seq_to_triples(row) for row in best]


def run_semantic_evaluation(predicted_graphs_lbl, title):
    gt_graphs_lbl = post_process_data(train_g, i2e, i2r)
    verifier      = SynPathsVerifier()                     
    evaluator     = SemanticEvaluator(
        predicted_graphs_lbl, gt_graphs_lbl,
        rule_checker    = verifier.check_rules_for_graph,
        entity_labels   = i2e,
        relation_labels = i2r
    )

    if not hasattr(evaluator, "organized_results"):
        if   hasattr(evaluator, "organize_results"):  evaluator.organize_results()
        elif hasattr(evaluator, "_organize_results"): evaluator._organize_results()
        else:                                         evaluator.evaluate_graphs()

    print(f"\nSemantic evaluation – {title}:")
    evaluator.print_results()
    return evaluator


#conditioned on test set generation
model.eval()
num_graphs       = 1000
generated_graphs = []

with torch.no_grad():
    for triples, _ in DataLoader(GraphSeqDataset(test_g), batch_size):
        generated_graphs.extend(
            generate(model, triples.to(device), beam_width)
        )
        if len(generated_graphs) >= num_graphs:
            generated_graphs = generated_graphs[:num_graphs]
            break

print("\nExample graph (conditioned on test entities):")
print(ints_to_labels(generated_graphs)[0])

run_semantic_evaluation(
    ints_to_labels(generated_graphs),
    title="graphs conditioned on test entities"
)

# strict_test = strict_graph_accuracy(
#     generated_graphs,
#     [list(map(tuple, t)) for t in
#      DataLoader(GraphSeqDataset(test_g), batch_size=batch_size)
#         .__iter__().__next__()[0].numpy()]
# )
# print(f"\nStrict graph acc. on first batch of test set: {strict_test:5.2f}%")


#sampled from latent space
num_samples   = 1000
z_rand        = torch.randn(num_samples, latent_dim, device=device)
latent_graphs = decode_latent(model, z_rand, beam=1)

print("\nExample graph (random latent):")
print(ints_to_labels(latent_graphs)[0])

run_semantic_evaluation(
    ints_to_labels(latent_graphs),
    title="graphs from random latent"
)




# #latent space exploration
# z_samples = torch.randn((100, latent_dim), device=device)
# z0 = z_samples[0].clone().detach()  
# latent_dim = z0.shape[0]
# n_directions = 20
# epsilon = 1.2 


# directions = torch.randn((n_directions, latent_dim))
# directions = directions / directions.norm(dim=1, keepdim=True)  


# directions = directions.to(z0.device)
# perturbed_zs = z0.unsqueeze(0) + epsilon * directions 
# perturbed_zs = perturbed_zs.to(device)

# decoded_graphs = decode_latent(model, perturbed_zs, beam=1)

# decoded_triples = ints_to_labels(decoded_graphs)

# ref_graph = decode_latent(model, z0.unsqueeze(0).to(device), beam=1)
# ref_triples = ints_to_labels(ref_graph)[0]

# print("\n=== Local Latent Neighborhood Exploration ===")
# print("\n--- Reference Graph (z₀) ---")
# for h, r, t in ref_triples:
#     print(f"({h}, {r}, {t})")

# for i, graph in enumerate(decoded_triples):
#     print(f"\n--- Perturbed z #{i+1} ---")
#     for h, r, t in graph:
#         print(f"({h}, {r}, {t})")


#     overlap = set(ref_triples) & set(graph)
#     print(f"# Overlapping triples with z₀: {len(overlap)} / {len(ref_triples)}")








#diversity metric
def canonical_graph_string(graph):
    return str(sorted(graph))

@torch.no_grad()
def count_unique_graphs(model, num_samples=1000, beam=1):
    model.eval()
    latent_dim = model.enc.mu.out_features  
    z_samples = torch.randn((num_samples, latent_dim), device=device)

    decoded_graphs = decode_latent(model, z_samples, beam=beam)
    graph_strings = [canonical_graph_string(g) for g in decoded_graphs]

    unique_graphs = set(graph_strings)
    print(f"\n[Graph Diversity from {num_samples} Random Latents]")
    print(f"  Unique graphs generated: {len(unique_graphs)}")
    print(f"  Diversity ratio: {len(unique_graphs) / num_samples:.3f}")

    return unique_graphs

_ = count_unique_graphs(model, num_samples=10000, beam=1)







import math, torch, numpy as np, torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

LN2 = math.log(2)

@torch.no_grad()
def ar_bits_sum(model, seq, z, pad_id=0):

    seq = seq.unsqueeze(0).to(z.device)        # (1, L)
    total_bits = 0.0
    for t in range(1, seq.size(1)):           
        target = seq[0, t].item()
        if target == pad_id:
            break                              
        logits = model.dec(z, seq[:, :t])[:, -1]
        log_prob = F.log_softmax(logits, dim=-1)[0, target]
        total_bits += -log_prob.item() / LN2
    return total_bits                         



@torch.no_grad()
def posterior_bits_per_graph(model, dataset, device, pad_id=0):

    records = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for triples, seq in tqdm(loader, desc="posterior bits-per-graph"):
        triples = triples.to(device)         # (1, T, 3)
        seq     = seq[0].to(device)          # (L,)


        z, mu, logv = model.enc(triples)

        kl_nats = -0.5 * torch.sum(1 + logv - mu.pow(2) - logv.exp(), dim=1)
        kl_bits = kl_nats.item() / LN2

        ar_bits = ar_bits_sum(model, seq, z, pad_id)

        records.append({
            "ar_bits":    ar_bits,
            "kl_bits":    kl_bits,
            "total_bits": ar_bits + kl_bits,
        })

    return records


@torch.no_grad()
def prior_bits_per_graph(model, num_samples, device, beam=1, pad_id=0):
    latent_dim = model.enc.mu.out_features
    z_samples  = torch.randn(num_samples, latent_dim, device=device)

    graphs     = decode_latent(model, z_samples, beam=beam)
    seqs       = [triples_to_seq(g).to(device) for g in graphs]

    bits = []
    for i in range(num_samples):
        z   = z_samples[i:i+1]
        seq = seqs[i]
        bits.append(ar_bits_sum(model, seq, z, pad_id))
    return bits


device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

full_set = GraphSeqDataset(test_g, permute=False)
subset   = torch.utils.data.Subset(full_set,
                                   range(int(0.01 * len(full_set))))  

post_records = posterior_bits_per_graph(model, subset, device)

post_total = np.array([r["total_bits"] for r in post_records])
post_ar    = np.array([r["ar_bits"]   for r in post_records])
post_kl    = np.array([r["kl_bits"]   for r in post_records])

print("\nPosterior bits-per-graph (reconstruction, 1 % test)")
print(f"  avg total : {post_total.mean():6.2f} bits")
print(f"  avg AR    : {post_ar.mean():6.2f}")
print(f"  avg KL    : {post_kl.mean():6.2f}")
print(f"  min / max : {post_total.min():.2f} / {post_total.max():.2f}")


# num_gen = len(subset)                       
# prior_total = np.array(
#     prior_bits_per_graph(model, num_gen, device, beam=1)
# )

# print("\nPrior bits-per-graph (generated graphs, matched z)")
# print(f"  avg total : {prior_total.mean():6.2f} bits")
# print(f"  min / max : {prior_total.min():.2f} / {prior_total.max():.2f}")







# # SIMPLE PROGRESSIVE TOKEN CONDITIONING 

# import numpy as np
# from torch.utils.data import DataLoader


# _NUM_PER_STEP = 10_000        
# _BEAM         = beam_width    
# _COND_GRAPHS  = test_g        
# _VERBOSE      = True

# def _safe_seq_to_triples(seq_tensor):
#     seq = seq_tensor.tolist()
#     triples, i = [], 1  # skip BOS
#     L = len(seq)
#     while i + 2 < L:
#         if seq[i] == SPECIAL["EOS"]:
#             break
#         h_tok, r_tok, t_tok = seq[i], seq[i+1], seq[i+2]
#         if not (ENT_BASE <= h_tok < REL_BASE):    break
#         if not (REL_BASE <= r_tok < VOCAB_SIZE):  break
#         if not (ENT_BASE <= t_tok < REL_BASE):    break
#         h, r, t = h_tok - ENT_BASE, r_tok - REL_BASE, t_tok - ENT_BASE
#         if not (0 <= h < num_entities and 0 <= r < num_relations and 0 <= t < num_entities):
#             break
#         triples.append((h,r,t))
#         i += 3
#     return triples

# def _safe_graphs_to_labels(graphs_int):
#     out = []
#     for g in graphs_int:
#         lbl_g = []
#         for (h,r,t) in g:
#             if 0 <= h < num_entities and 0 <= r < num_relations and 0 <= t < num_entities:
#                 lbl_g.append((i2e[h], i2r[r], i2e[t]))
#         out.append(lbl_g)
#     return out


# def _diversity(graphs_int):
#     cg = [canonical_graph_string(g) for g in graphs_int]  
#     uniq = len(set(cg))
#     return uniq, (uniq / len(graphs_int) if graphs_int else float('nan'))


# def _slot_type_from_id(tok_id: int):
#     if ENT_BASE <= tok_id < REL_BASE:     return 'E'
#     if REL_BASE <= tok_id < VOCAB_SIZE:   return 'R'
#     return 'X'


# _ref_graph = train_g[0]                       
# _ref_seq   = triples_to_seq(_ref_graph)       
# _ref_np    = _ref_seq.cpu().numpy()

# _eos_pos   = np.where(_ref_np == SPECIAL["EOS"])[0]
# _eos_pos   = int(_eos_pos[0]) if len(_eos_pos) else len(_ref_np)
# _anchor_ids = _ref_np[1:_eos_pos]            
# _anchor_types = [_slot_type_from_id(int(t)) for t in _anchor_ids]

# if _VERBOSE:
#     print("\n[progressive conditioning] reference graph:", _ref_graph)
#     print("[progressive conditioning] anchor token IDs :", _anchor_ids.tolist())
#     print("[progressive conditioning] slot types       :", _anchor_types)

# @torch.no_grad()
# def _constrained_generate(model, triples, constraints, beam=_BEAM):
#     dev = next(model.parameters()).device
#     B   = triples.size(0)
#     z, *_ = model.enc(triples.to(dev))

#     BOS  = torch.full((B,1), SPECIAL["BOS"], device=dev, dtype=torch.long)
#     seqs = [(BOS, torch.zeros(B, device=dev))]

#     for step in range(seq_len-1):
#         cand = []
#         for s, lp in seqs:
#             logits = model.dec(z, s)[:, -1]         
#             logp   = F.log_softmax(logits, dim=-1)

#             if step in constraints:                  
#                 forced = constraints[step]
#                 mask = torch.full_like(logp, float('-inf'))
#                 mask[:, forced] = logp[:, forced]
#                 logp = mask

#             top_lp, top_id = logp.topk(beam, dim=-1)
#             for k in range(beam):
#                 cand.append((torch.cat([s, top_id[:,k,None]], 1),
#                              lp + top_lp[:,k]))
#         seqs = sorted(cand, key=lambda x: x[1].mean().item(), reverse=True)[:beam]
#         if all((s[:,-1] == SPECIAL["EOS"]).all() for s,_ in seqs):
#             break

#     best = seqs[0][0].cpu()
#     return [_safe_seq_to_triples(row) for row in best]

# _pc_loader = DataLoader(
#     GraphSeqDataset(_COND_GRAPHS, triple_order="keep", permute=False),
#     batch_size=batch_size,
#     shuffle=True,
#     drop_last=False
# )

# def _gen_under(active_constraints):
#     out = []
#     with torch.no_grad():
#         for triples, _ in _pc_loader:
#             triples = triples.to(device)
#             out.extend(_constrained_generate(model, triples, active_constraints))
#             if len(out) >= _NUM_PER_STEP:
#                 out = out[:_NUM_PER_STEP]
#                 break
#     return out

# _forward_stats = []
# _active = {}
# for idx, tok_id in enumerate(_anchor_ids):
#     _active[idx] = int(tok_id)        
#     stype = _anchor_types[idx]

#     print(f"\n--- forward step {idx+1}/{len(_anchor_ids)} "
#           f"| constrain slot {idx} ({'entity' if stype=='E' else 'relation'}) ---")

#     gen = _gen_under(_active)
#     print(f"generated: {len(gen)}")

#     _ = run_semantic_evaluation(_safe_graphs_to_labels(gen),
#                                 title=f"forward step {idx+1}")

#     uniq, ratio = _diversity(gen)
#     print(f"diversity: {uniq} unique / {len(gen)} ({ratio:.3f})")

#     _forward_stats.append(dict(step=idx+1, slot=idx, type=stype,
#                                n=len(gen), uniq=uniq, ratio=ratio))

