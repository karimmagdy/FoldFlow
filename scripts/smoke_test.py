"""Quick smoke test for speed patches."""
import sys
sys.path.insert(0, ".")
from foldflow.models.classifier import FoldFlowClassifier
from foldflow.utils.training import EMA
import torch

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = FoldFlowClassifier(num_classes=10, hidden_dim=256, num_particles=64, particle_dim=64, num_steps=8, encoder_type="weak").to(device)
ema = EMA(model, decay=0.999)
x = torch.randn(4, 3, 32, 32, device=device)

# Training mode - should return 1 aux_logit (only last step)
model.train()
result = model(x)
print(f"Logits shape: {result['logits'].shape}")
print(f"Aux logits count: {len(result.get('aux_logits', []))} (expected 1)")
assert len(result.get("aux_logits", [])) == 1, "Should have exactly 1 aux_logit"

# Eval mode - should return no aux_logits
model.eval()
with torch.no_grad():
    result = model(x)
    print(f"Eval aux_logits: {len(result.get('aux_logits', []))} (expected 0)")

# EMA update
model.train()
ema.update(model)
print("EMA update OK")
print("All checks passed!")
