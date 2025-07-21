
import os
import math, torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from intelligraphs.data_loaders import load_data_as_list
from tqdm import tqdm
import wandb

from intelligraphs.data_loaders import load_data_as_list
from intelligraphs.evaluators   import post_process_data, SemanticEvaluator



total_epochs   = 151
batch_size     = 16          
d_model        = 512
latent_dim     = 128
nhead          = 8
num_layers     = 6
beam_width     = 4
eval_every     = 9999


dataset_name = 'wd-articles'
wandb.init(
    project="autoreg_articles",
    name="Autoreg" + dataset_name,
    config=dict(
        total_epochs=total_epochs,
        batch_size=batch_size,
        d_model=d_model,
        latent_dim=latent_dim,
        nhead=nhead,
        num_layers=num_layers,
        beam_width=beam_width,
        dataset=dataset_name,
        lr=1e-4,
        beta0=0.1,
        beta1=1.0,
    ),
)


USE_CHECKPOINTS  = False                
CHECKPOINT_PATH  = "model_checkpoint_articles.pt"


dataset_name = "wd-articles"               
(train_g, val_g, test_g,
 (e2i, i2e),
 (r2i, i2r),
 (min_edges, max_edges), _) = load_data_as_list(dataset_name)


RAW_NUM_ENTITIES  = len(e2i)
RAW_NUM_RELATIONS = len(r2i)

PAD_EID = RAW_NUM_ENTITIES           
PAD_RID = RAW_NUM_RELATIONS          

num_entities  = RAW_NUM_ENTITIES  + 1
num_relations = RAW_NUM_RELATIONS + 1
max_triples   = max_edges

print(f"entities: {RAW_NUM_ENTITIES} (+PAD)  "
      f"relations: {RAW_NUM_RELATIONS} (+PAD)  "
      f"max_triples: {max_triples}")

SPECIAL      = {"PAD":0, "BOS":1, "EOS":2}
ENT_BASE     = 3
REL_BASE     = ENT_BASE + RAW_NUM_ENTITIES
VOCAB_SIZE   = REL_BASE + RAW_NUM_RELATIONS
seq_len      = 1 + max_triples*3 + 1

def triples_to_seq(triples):
    seq = [SPECIAL["BOS"]]
    for h,r,t in triples:
        seq += [ENT_BASE+h, REL_BASE+r, ENT_BASE+t]
    seq.append(SPECIAL["EOS"])
    seq += [SPECIAL["PAD"]] * (seq_len - len(seq))
    return torch.tensor(seq, dtype=torch.long)

def seq_to_triples(seq):
    triples, i = [], 1
    while i + 2 < len(seq) and seq[i] != SPECIAL["EOS"]:
        h, r, t = seq[i:i+3].tolist()
        triples.append((h - ENT_BASE, r - REL_BASE, t - ENT_BASE))
        i += 3
    return triples


def pad_triples(triples):
    pad = (PAD_EID, PAD_RID, PAD_EID)
    return torch.tensor(triples + [pad] * (max_triples - len(triples)),
                        dtype=torch.long)

class GraphSeqDataset(Dataset):
    def __init__(self, graphs): self.graphs = graphs
    def __len__(self): return len(self.graphs)
    def __getitem__(self, idx):
        triples = self.graphs[idx]
        return pad_triples(triples), triples_to_seq(triples)


class GraphEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.e_emb = nn.Embedding(num_entities,  d_model, padding_idx=PAD_EID)
        self.r_emb = nn.Embedding(num_relations, d_model, padding_idx=PAD_RID)
        self.txf   = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model*3, nhead, batch_first=True), 2)
        self.mu, self.logv = nn.Linear(d_model*3, latent_dim), nn.Linear(d_model*3, latent_dim)

    def forward(self, triples):                    
        mask = triples[:, :, 1] != PAD_RID        
        h = self.e_emb(triples[:, :, 0])
        r = self.r_emb(triples[:, :, 1])
        t = self.e_emb(triples[:, :, 2])

        x = torch.cat([h, r, t], -1)
        x = self.txf(x, src_key_padding_mask=~mask)   
        x = (x * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)

        mu, logv = self.mu(x), self.logv(x)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * logv)
        return z, mu, logv

class AutoRegDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_emb = nn.Embedding(seq_len,   d_model)
        self.z_proj  = nn.Linear(latent_dim,   d_model)
        layer = nn.TransformerDecoderLayer(d_model, nhead, batch_first=True)
        self.txf = nn.TransformerDecoder(layer, num_layers)
        self.out = nn.Linear(d_model, VOCAB_SIZE)
    def forward(self, z, tgt):
        B, L = tgt.shape
        tok = self.tok_emb(tgt)
        pos = self.pos_emb(torch.arange(L, device=z.device)).unsqueeze(0)
        mem = self.z_proj(z).unsqueeze(1).repeat(1, L, 1)
        mask = torch.triu(torch.ones(L, L, device=z.device, dtype=torch.bool), 1)
        return self.out(self.txf(tok + pos, mem, tgt_mask=mask))

class Model(nn.Module):
    def __init__(self): super().__init__(); self.enc, self.dec = GraphEncoder(), AutoRegDecoder()
    def forward(self, triples, seq_in):
        z, mu, logv = self.enc(triples)
        return self.dec(z, seq_in), mu, logv


@torch.no_grad()
def generate(model, triples, beam=beam_width):
    device = next(model.parameters()).device
    B = triples.size(0)
    z, _, _ = model.enc(triples.to(device))
    BOS = torch.full((B, 1), SPECIAL["BOS"], device=device)
    seqs = [(BOS, torch.zeros(B, device=device))]
    for _ in range(seq_len - 1):
        cand = []
        for s, lp in seqs:
            logp = F.log_softmax(model.dec(z, s)[:, -1], -1)
            top_lp, top_id = logp.topk(beam, -1)
            for k in range(beam):
                cand.append((torch.cat([s, top_id[:, k, None]], 1), lp + top_lp[:, k]))
        seqs = sorted(cand, key=lambda x: x[1].mean().item(), reverse=True)[:beam]
        if all((s[:, -1] == SPECIAL["EOS"]).all() for s, _ in seqs): break
    best = seqs[0][0].cpu()
    return [seq_to_triples(s) for s in best]

def strict_graph_accuracy(pred, gt):
    return 100. * sum(set(p) == set(g) for p, g in zip(pred, gt)) / len(gt)

def kl(mu, logv): return -0.5 * torch.mean(1 + logv - mu.pow(2) - logv.exp())


train_loader = DataLoader(GraphSeqDataset(train_g), batch_size,
                          shuffle=False, drop_last=True)
val_loader   = DataLoader(GraphSeqDataset(val_g),   batch_size)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = Model().to(device)
opt    = torch.optim.Adam(model.parameters(), 1e-4)
sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_epochs, eta_min=1e-6)
wandb.watch(model, log="all", log_freq=100)


def cyclical_beta(epoch, M=4, R=0.5):
    cycle = epoch % (total_epochs // M)
    frac  = cycle / (total_epochs // M)
    return min(1.0, frac / R)

start_epoch = 0
if USE_CHECKPOINTS and os.path.exists(CHECKPOINT_PATH):
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    opt.load_state_dict(checkpoint["optimizer_state_dict"])
    sched.load_state_dict(checkpoint["scheduler_state_dict"])
    start_epoch = checkpoint.get("epoch", 0)
    print(f"Loaded checkpoint from epoch {start_epoch}")

total_losses, ce_losses, kl_losses, strict_acc = [], [], [], []
beta0, beta1 = 0.1,1.0
for epoch in range(start_epoch, total_epochs):
    model.train()
    beta = beta0 + (beta1 - beta0) * epoch / total_epochs

    ce_sum = kl_sum = tot_sum = 0; batches = 0

    for triples, seq in train_loader:
        triples, seq = triples.to(device), seq.to(device)
        logits, mu, logv = model(triples, seq[:, :-1])
        ce = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE),
                             seq[:, 1:].reshape(-1),
                             ignore_index=SPECIAL["PAD"])
        kl_val = kl(mu, logv)
        loss   = ce + beta * kl_val
        opt.zero_grad(); loss.backward(); opt.step()
        ce_sum += ce.item(); kl_sum += kl_val.item(); tot_sum += loss.item(); batches += 1
    sched.step()

    ce_epoch, kl_epoch, total_epoch = ce_sum / batches, kl_sum / batches, tot_sum / batches
    print(f"Epoch {epoch+1:3d}/{total_epochs} | "
          f"Total {total_epoch:8.4f} | CE {ce_epoch:8.4f} | KL {kl_epoch:8.4f}")

    total_losses.append(total_epoch); ce_losses.append(ce_epoch); kl_losses.append(kl_epoch)

    wandb.log({
    "train/loss_total": total_epoch,
    "train/loss_ce": ce_epoch,
    "train/loss_kl": kl_epoch,
    "train/beta": beta,
    "train/lr": sched.get_last_lr()[0],
    }, step=epoch)

    if epoch % 5 == 0:
        torch.save({
            "epoch":                epoch + 1,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sched.state_dict()
        }, CHECKPOINT_PATH)
        print(f"Saved checkpoint at epoch {epoch+1}")
    




try:
    from intelligraphs.verifier.wikidata.wdarticles_verifier import WDArticlesVerifier
    # from intelligraphs.verifier.wikidata.wdmovies_verifier import WDMoviesVerifier
except ImportError:
    # fallback for older package versions
    # from intelligraphs.verifier.wd_movies_verifier import WDMoviesVerifier
    from intelligraphs.verifier.wdarticles_verifier import WDArticlesVerifier


(train_g, val_g, test_g,
 (e2i, i2e),
 (r2i, i2r),
 *_ ) = load_data_as_list("wd-articles")


def ints_to_labels(graphs):
    result = []
    skipped = 0
    for g in graphs:
        clean_graph = []
        for h, r, t in g:
            if h in i2e and r in i2r and t in i2e:
                clean_graph.append((i2e[h], i2r[r], i2e[t]))
            else:
                skipped += 1
        result.append(clean_graph)
    if skipped > 0:
        print(f"[!] Skipped {skipped} invalid triples")
    return result

@torch.no_grad()
def decode_latent(model, z, beam=beam_width):
    z      = z.to(next(model.parameters()).device, dtype=torch.float32)
    B      = z.size(0)
    BOS    = torch.full((B, 1), SPECIAL["BOS"], dtype=torch.long, device=z.device)
    seqs   = [(BOS, torch.zeros(B, device=z.device))]

    for _ in range(seq_len - 1):
        cand = []
        for s, lp in seqs:
            logits      = model.dec(z, s)[:, -1]
            logp        = torch.nn.functional.log_softmax(logits, dim=-1)
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
    verifier      = WDArticlesVerifier()
    evaluator     = SemanticEvaluator(
        predicted_graphs_lbl, gt_graphs_lbl,
        rule_checker   = verifier.check_rules_for_graph,
        entity_labels  = i2e,
        relation_labels= i2r
    )

    if not hasattr(evaluator, "organized_results"):
        if   hasattr(evaluator, "organize_results"):   evaluator.organize_results()
        elif hasattr(evaluator, "_organize_results"): evaluator._organize_results()
        elif hasattr(evaluator, "evaluate_graphs"):   evaluator.evaluate_graphs()

    print(f"\nSemantic evaluation – {title}:")
    evaluator.print_results()
    return evaluator


model.eval()
num_graphs       = 1000
generated_graphs = []

test_loader = DataLoader(GraphSeqDataset(test_g), batch_size)

with torch.no_grad():
    for triples, _ in test_loader:
        generated_graphs.extend(
            generate(model, triples=triples.to(device), beam=beam_width)
        )
        if len(generated_graphs) >= num_graphs:
            generated_graphs = generated_graphs[:num_graphs]
            break

print("\nExample graph (conditioned on test entities):")
print(ints_to_labels(generated_graphs)[0])
table_cond = wandb.Table(columns=["graph"])
for g in ints_to_labels(generated_graphs[:5]):
    table_cond.add_data(str(g))
wandb.log({"samples/conditioned_on_test": table_cond})


run_semantic_evaluation(
    ints_to_labels(generated_graphs),
    title="graphs conditioned on test entities"
)


num_samples   = 1000
z_rand        = torch.randn(num_samples, latent_dim, device=device)
latent_graphs = decode_latent(model, z_rand, beam=1)

print("\nExample graph (random latent):")
print(ints_to_labels(latent_graphs)[0])

run_semantic_evaluation(
    ints_to_labels(latent_graphs),
    title="graphs from random latent"
)

table = wandb.Table(columns=["graph"])
for g in ints_to_labels(latent_graphs[:5]):
    table.add_data(str(g))
wandb.log({"samples/random_latent": table})





def canonical_graph_string(graph):
    return str(sorted(graph))

@torch.no_grad()
def count_unique_graphs(model, num_samples=10, beam=1):
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

div = count_unique_graphs(model, num_samples=10, beam=1)
wandb.log({
    "diversity/unique_graphs": len(div),
    "diversity/ratio": len(div) / 10,
})






@torch.no_grad()
def bits_per_sequence(model, seq, z, pad_id=0):
    seq = seq.unsqueeze(0).to(z.device)  
    total = 0.0
    for t in range(1, seq.size(1)):  
        target = seq[0, t].item()
        if target == pad_id:
            break
        logits = model.dec(z, seq[:, :t])[:, -1]  # shape: (1, vocab_size)
        log_probs = F.log_softmax(logits, dim=-1)
        total += -log_probs[0, target].item() / math.log(2)  # convert to bits
    return total

@torch.no_grad()
def posterior_bits_full(model, dataset, device, pad_id=0):
    records = []
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    for triples, seq in tqdm(loader, desc="Posterior full evaluation"):
        triples = triples.to(device)     
        seq     = seq[0].to(device)      

        z, mu, logv = model.enc(triples)  


        ar_bits = bits_per_sequence(model, seq, z, pad_id)

        kl_nats = -0.5 * torch.sum(1 + logv - mu.pow(2) - logv.exp(), dim=1) 
        kl_bits = kl_nats.item() / math.log(2)

        records.append({
            "ar_bits": ar_bits,
            "kl_bits": kl_bits,
            "total_bits": ar_bits + kl_bits,
            "z": z.squeeze(0).cpu().numpy(),
            "mu": mu.squeeze(0).cpu().numpy(),
            "logv": logv.squeeze(0).cpu().numpy(),
        })

    return records

subset = torch.utils.data.Subset(GraphSeqDataset(test_g), range(int(0.01 * len(test_g))))
results = posterior_bits_full(model, subset, device)

total_bits = [r["total_bits"] for r in results]
ar_bits    = [r["ar_bits"] for r in results]
kl_bits    = [r["kl_bits"] for r in results]

print("\n[Full Compression (Posterior) on 1% of Test Set]")
print(f"  Avg total bits: {np.mean(total_bits):.2f}")
print(f"  Avg AR bits:    {np.mean(ar_bits):.2f}")
print(f"  Avg KL bits:    {np.mean(kl_bits):.2f}")
print(f"  Min / Max total bits: {np.min(total_bits):.2f} / {np.max(total_bits):.2f}")


wandb.log({
    "compression/avg_total_bits": np.mean(total_bits),
    "compression/avg_ar_bits": np.mean(ar_bits),
    "compression/avg_kl_bits": np.mean(kl_bits),
})
