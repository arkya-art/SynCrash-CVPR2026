import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class BinaryAccidentModel(nn.Module):
    def __init__(self, nexar_weights_path=None, freeze_backbone=False):
        super().__init__()
        
        print("Loading VideoMAEv2-giant config...")
        config = AutoConfig.from_pretrained("OpenGVLab/VideoMAEv2-giant", trust_remote_code=True)
        config.drop_path_rate = 0.1 # Stochastic depth for large model regularisation
        
        print("Initializing model backbone...")
        self.backbone = AutoModel.from_config(config, trust_remote_code=True)
        
        # Metadata Embeddings based on METADATA_VOCABS lengths from dataset script
        # scene_layout (8 classes), weather (4 classes), day_time (3 classes)
        self.scene_emb = nn.Embedding(10, 32)
        self.weather_emb = nn.Embedding(6, 32)
        self.day_time_emb = nn.Embedding(5, 32)
        
        embed_dim = getattr(config, "embed_dim", getattr(config, "hidden_size", 1408))
        print(f"[MODEL] Backbone hidden_size (embed_dim): {embed_dim}")
        
        # Single Classification Head
        self.fc = nn.Sequential(
            nn.Linear(embed_dim + 32 + 32 + 32, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )
        
        if nexar_weights_path:
            print(f"Loading Nexar winning weights from: {nexar_weights_path}")
            state_dict = torch.load(nexar_weights_path, map_location='cpu')
            
            # The Nexar best.pth is a training checkpoint dict:
            # {'epoch': N, 'model': {weights...}, 'optimizer': ..., ...}
            # We must extract the actual weights from the 'model' key.
            if isinstance(state_dict, dict) and 'model' in state_dict:
                print("  Detected checkpoint wrapper, extracting 'model' sub-dict...")
                state_dict = state_dict['model']
            
            # Strip common prefixes
            clean_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    k = k.replace("module.", "", 1)
                if k.startswith("backbone."):
                    k = k.replace("backbone.", "", 1)
                clean_state_dict[k] = v
                
            missing, unexpected = self.backbone.load_state_dict(clean_state_dict, strict=False)
            print(f"Loaded Nexar Backbone. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
            if missing:
                print(f"  Top-5 Missing keys: {missing[:5]}")
            if unexpected:
                print(f"  Top-5 Unexpected keys: {unexpected[:5]}")
            
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Backbone frozen for training head only.")
        else:
            self.backbone.with_cp = True
                
    def forward(self, pixel_values, scene_idx=None, weather_idx=None, day_time_idx=None):
        # Forward pass through VideoMAE backbone
        # OpenGVLab VideoMAEv2 returns a raw Tensor, not a HuggingFace output object
        outputs = self.backbone(pixel_values=pixel_values)
        
        # Handle both HuggingFace output objects AND raw tensor returns
        if hasattr(outputs, 'last_hidden_state'):
            vid_feats = outputs.last_hidden_state.mean(dim=1)  # [B, hidden]
        elif isinstance(outputs, torch.Tensor):
            if outputs.dim() == 3:   # [B, SeqLen, Hidden]
                vid_feats = outputs.mean(dim=1)
            else:                     # [B, Hidden] already pooled
                vid_feats = outputs
        else:
            # Some models return tuples
            vid_feats = outputs[0].mean(dim=1)
        
        B = vid_feats.size(0)
        device = vid_feats.device
        
        # Inject Metdata or zeros
        if scene_idx is not None and weather_idx is not None and day_time_idx is not None:
            s_emb = self.scene_emb(scene_idx)
            w_emb = self.weather_emb(weather_idx)
            d_emb = self.day_time_emb(day_time_idx)
        else:
            s_emb = torch.zeros(B, 32, device=device)
            w_emb = torch.zeros(B, 32, device=device)
            d_emb = torch.zeros(B, 32, device=device)
            
        # Concatenate visual temporal features + environmental metadata
        combined = torch.cat([vid_feats, s_emb, w_emb, d_emb], dim=1)
        
        # Output logit
        logits = self.fc(combined) # [B, 1]
        
        return logits.squeeze(-1) # [B]
