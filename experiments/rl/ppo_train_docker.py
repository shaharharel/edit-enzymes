"""PPO training step inside Docker. Called by host orchestrator each round.

Loads model from /ckpt/model.pt if exists, runs PPO update, saves back.
Proper differentiable PPO loss through RFD3's training_step.

Args via command line:
    --cif-advantages JSON string of [[cif_path, advantage], ...]
    --ppo-epochs Number of PPO epochs
    --clip-epsilon PPO clip epsilon
    --lr Learning rate
"""
import torch, sys, os, json, numpy as np, argparse, time

parser = argparse.ArgumentParser()
parser.add_argument('--cif-advantages', type=str, required=True)
parser.add_argument('--ppo-epochs', type=int, default=3)
parser.add_argument('--clip-epsilon', type=float, default=0.2)
parser.add_argument('--lr', type=float, default=1e-5)
args = parser.parse_args()

cif_advantages = json.loads(args.cif_advantages)

from atomworks.io.parser import parse, initialize_chain_info_from_atom_array
from rfd3.transforms.conditioning_base import REQUIRED_CONDITIONING_ANNOTATIONS
from rfd3.engine import RFD3InferenceConfig, RFD3InferenceEngine
from rfd3.metrics.losses import DiffusionLoss, SequenceLoss

# Load engine
conf = RFD3InferenceConfig(ckpt_path="/root/.foundry/checkpoints/rfd3_latest.ckpt",
    diffusion_batch_size=1, inference_sampler={"num_timesteps": 3})
engine = RFD3InferenceEngine(**conf)
engine._set_out_dir("/tmp/ppo")
engine.initialize()

# Loss
class L(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.d = DiffusionLoss(weight=4.0,sigma_data=16.0,lddt_weight=0.25,alpha_virtual_atom=1.0,
            alpha_polar_residues=1.0,alpha_ligand=10.0,lp_weight=0.0,unindexed_norm_p=1.0,
            alpha_unindexed_diffused=1.0,unindexed_t_alpha=0.75)
        self.s = SequenceLoss(weight=0.1,max_t=1)
    def forward(self,**kw):
        dl,dd=self.d(**kw); sl,sd=self.s(**kw); return dl+sl,{**dd,**sd}

engine.trainer.loss = L().cuda()
model = engine.trainer.state["model"]
opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=args.lr)

# LOAD PERSISTED CHECKPOINT (key fix!)
if os.path.exists("/ckpt/model.pt"):
    saved = torch.load("/ckpt/model.pt", map_location="cuda", weights_only=False)
    model.load_state_dict(saved, strict=False)
    print("Loaded persisted weights", flush=True)

# Parse CIFs
def load_cif(path):
    r = parse(path)
    aa = r["asym_unit"][0]
    initialize_chain_info_from_atom_array(aa)
    aa.set_annotation("chain_iid", aa.chain_id.copy())
    if "pn_unit_iid" not in aa.get_annotation_categories():
        aa.set_annotation("pn_unit_iid", aa.pn_unit_id.copy())
    for ann in REQUIRED_CONDITIONING_ANNOTATIONS:
        if ann not in aa.get_annotation_categories():
            aa.set_annotation(ann, np.zeros(len(aa), dtype=bool))
    return aa

pipeline_data = []
for cif_path, advantage in cif_advantages:
    if not os.path.exists(cif_path): continue
    try:
        aa = load_cif(cif_path)
        pr = engine.pipeline({"atom_array": aa, "example_id": os.path.basename(cif_path)})
        pr = engine.trainer.fabric.to_device(pr)
        pipeline_data.append((pr, advantage))
    except Exception as e:
        print(f"Skip {cif_path}: {e}", flush=True)

if not pipeline_data:
    print("PPO_RESULT: n=0 loss=0.0", flush=True)
    sys.exit(0)

# STEP 1: Compute old losses (no grad)
model.eval()
old_losses = []
for pr, _ in pipeline_data:
    with torch.no_grad():
        ni = engine.trainer._assemble_network_inputs(pr)
        no = model.forward(input=ni, n_cycle=0)
        li = engine.trainer._assemble_loss_extra_info(pr)
        lv, _ = engine.trainer.loss(network_input=ni, network_output=no, loss_input=li)
        old_losses.append(lv.item())

# STEP 2: PPO epochs with PROPER differentiable loss
model.train()
total_loss = 0
n_updates = 0

for epoch in range(args.ppo_epochs):
    for idx, (pr, advantage) in enumerate(pipeline_data):
        opt.zero_grad()

        # New forward pass WITH gradients
        ni = engine.trainer._assemble_network_inputs(pr)
        no = model.forward(input=ni, n_cycle=0)
        li = engine.trainer._assemble_loss_extra_info(pr)
        new_loss, _ = engine.trainer.loss(network_input=ni, network_output=no, loss_input=li)
        # new_loss IS a tensor with grad

        # PPO ratio (differentiable through new_loss)
        log_ratio = old_losses[idx] - new_loss  # old is detached scalar, new is tensor
        ratio = torch.exp(torch.clamp(log_ratio, -10, 10))

        # Clipped surrogate
        adv = torch.tensor(advantage, device=ratio.device)
        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1 - args.clip_epsilon, 1 + args.clip_epsilon) * adv
        ppo_loss = -torch.min(surr1, surr2)

        # Backward through PPO loss
        ppo_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step()

        total_loss += ppo_loss.item()
        n_updates += 1

avg_loss = total_loss / max(n_updates, 1)
print(f"PPO_RESULT: n={n_updates} loss={avg_loss:.4f}", flush=True)

# SAVE CHECKPOINT (key fix!)
torch.save(model.state_dict(), "/ckpt/model.pt")
print("Checkpoint saved", flush=True)
