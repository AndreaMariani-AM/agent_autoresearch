import os
from pathlib import Path
import sys
# project_root = Path(__file__).parent.parent
# sys.path.insert(0, str(project_root))
sys.path.append(os.path.abspath('/group/glastonbury/andrea/projects/IBD/IBD_predictive_model/src')) 
from src.models.experts_MIL import ClassConditionalAdditiveMIL, ExpertOutput
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Literal, Dict
# from src.models.attention import AttnVanilla, GatedAttn
from models.modules import MLP
import lightning as L
from mammoth import Mammoth
from models.attention import GatedAttn
# from models.mamba import SRMamba

def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

# Main MIL model class
class DiscriminatorNet(nn.Module):
    """
    Discriminator network for MIL.
    """
    def __init__(
            self,
            input_dim: int = 2560,
            hidden_dim: int = 1280,
            output_dim: int = 256,
            n_classes: int = 1, #binary classification
            dropout: float = 0.25,
    ):
        """
        Args:
            instance_batch_size: Process instances in batches due to memory constrainsts
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # instance-level encoder
        self.MLP = MLP(
            in_features=input_dim,
            hidden_features=hidden_dim,
            out_features=output_dim,
            dropout=dropout,
            act_layer=nn.ReLU,
        )

        self.attn = GatedAttn(
            input_dim=output_dim,
            hidden_dim=output_dim // 2,
            n_classes=n_classes,
        )

        #Bag-level classifier
        self.cls = nn.Linear(output_dim, n_classes) # (output_dim,) --> (1,) for binary classification
        self.apply(initialize_weights)

    def forward(
            self,
            x: torch.Tensor,
            attn_return: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], None]:
        """
        Args:
            x: Tensor of shape (1, n_instances, n_features), a single bag of instances is a WSI
            attn_return: Whether to return attention weights
        Returns:
            logits: Tensor of shape (n_classes,)
            attn_weights: Optional tensor of shape (n_instances, n_classes)
            contributions: Always None for AttentionMIL (no per-instance predictions)
        """
        x = x.squeeze(0) #remove dataloader batch dimension

        h = self.MLP(x) # (n_instances, n_featues) --> (n_instances, output_dim)
        # Let's try with attention
        A = self.attn(h)
        A = F.softmax(A, dim=0) # (n_instances, n_classes)
        h_bag = (A * h).sum(dim=0) # (output_dim,)
        # Use bag representations to classify 
        logits = self.cls(h_bag) # (n_classes,)

        return logits
    
class MammothNet(nn.Module):
    """
    Parameters
    ----------
    """
    def __init__(
        self,
        input_dim: int = 2560,
        hidden_dim: int = 1280,
        output_dim: int = 256,
        num_classes: int = 1,
        dropout: float = 0.25,
        pooling: str = "sum",
        moe_args={}
    ):
        super().__init__()
        self.proj_dim = output_dim

        # ── Shared instance encoder: raw_dim → proj_dim ──────────────
        if moe_args and moe_args.get("num_experts", 0) > 0:
            self.encoder = Mammoth(**moe_args)
        else:
            self.encoder = MLP(
            in_features=input_dim,
            hidden_features=hidden_dim,
            out_features=output_dim,
            dropout=dropout,
            act_layer=nn.ReLU,
        )

        # ── Core additive MIL on projected patch embeddings ──────────
        self.mil = ClassConditionalAdditiveMIL(
            in_dim=output_dim,
            num_classes=num_classes,
            key_dim=output_dim // 2,
            dropout=dropout,
            pooling=pooling,
        )

        self.apply(initialize_weights)

    # ------------------------------------------------------------------
    def forward(
        self,
        h_patches: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[ExpertOutput, torch.Tensor]:
        """
        Parameters
        ----------
        h_patches : Tensor, shape ``(N_patches, D_raw)``
            Raw foundation-model embeddings for every 256×256 patch
            in a single WSI.  Already squeezed (no batch dim).
        mask : Tensor, shape ``(N_patches,)`` bool, optional
            ``True`` = valid patch.

        Returns
        -------
        expert_out : ExpertOutput
            Standardised expert output (logits, attn, contributions,
            bag representation).
        top_k_indices : Tensor, shape ``(K,)`` int64
            Indices (into the *original* patch array) of the top-k
            most-attended patches.  These are passed to CellExpert.
        """
        # 1) Project patches to hidden space ──────────────────────────
        h = self.encoder(h_patches)  # (N, proj_dim)
        
        if h.dim() == 3:
            h = h.squeeze(0)  # remove MoE expert dim if present (1, N, proj_dim) → (N, proj_dim)

        # 2) Additive MIL ─────────────────────────────────────────────
        logits, attn_w, contribs, bag_repr = self.mil(h, mask=mask)

        expert_out = ExpertOutput(
            logits=logits,
            attn_weights=attn_w,
            contributions=contribs,
            representation=bag_repr,
        )

        return expert_out

class MambaMoothNet(nn.Module):
    """
    This is a weird mix between Mammoth and MambaMIL
    ----------
    """
    def __init__(
        self,
        input_dim: int = 2560,
        hidden_dim: int = 1280,
        output_dim: int = 512,
        num_classes: int = 1,
        dropout: float = 0.25,
        mamba_layers=2, 
        rate=10,
        pooling: str = "sum",
        moe_args={}
    ):
        super().__init__()
        self.mamba_input_dim = output_dim
        self.rate = rate

        # ── Shared instance encoder: raw_dim → proj_dim ──────────────
        if moe_args and moe_args.get("num_experts", 0) > 0:
            self.encoder = Mammoth(**moe_args)
        else:
            self.encoder = MLP(
            in_features=input_dim,
            hidden_features=hidden_dim,
            out_features=output_dim,
            dropout=dropout,
            act_layer=nn.ReLU,
        )
            
        # set mammbaMIL layers
        self.mamba_layers = nn.ModuleList()
        for _ in range(mamba_layers):
            self.mamba_layers.append(
                nn.Sequential(
                    nn.LayerNorm(self.mamba_input_dim),
                    SRMamba(
                        d_model=self.mamba_input_dim,
                        d_state=16,
                        d_cov=4,
                        expand=2
                    ),
                )
            )

        # ── Core additive MIL on projected patch embeddings ──────────
        self.mil = ClassConditionalAdditiveMIL(
            in_dim=output_dim,
            num_classes=num_classes,
            key_dim=output_dim // 2,
            dropout=dropout,
            pooling=pooling,
        )

        self.apply(initialize_weights)

    # ------------------------------------------------------------------
    def forward(
        self,
        h_patches: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[ExpertOutput, torch.Tensor]:
        """
        Parameters
        ----------
        h_patches : Tensor, shape ``(N_patches, D_raw)``
            Raw foundation-model embeddings for every 256×256 patch
            in a single WSI.  Already squeezed (no batch dim).
        mask : Tensor, shape ``(N_patches,)`` bool, optional
            ``True`` = valid patch.

        Returns
        -------
        expert_out : ExpertOutput
            Standardised expert output (logits, attn, contributions,
            bag representation).
        """
        # 1) Project patches
        h = self.encoder(h_patches)  # (N, proj_dim)
        
        if h.dim() == 2:
            h = h.unsqueeze(0)  # add batch dim if missing (N, proj_dim) → (1, N, proj_dim)
        h = h.float()  # ensure float32 for MambaMIL layers

        # 2) MambaMIL layers
        for layer in self.mamba_layers:
            h_ = h
            h = layer[0](h)
            h = layer[1](h, rate=self.rate)
            h = h + h_  # residual connection
        
        # normalize h
        h = nn.LayerNorm(h.size(-1)).to(h.device)(h)
        h = h.squeeze(0)  # remove batch dim (1, N, proj_dim) → (N, proj_dim)

        # 2) Additive MIL ─────────────────────────────────────────────
        logits, attn_w, contribs, _ = self.mil(h, mask=mask)

        expert_out = ExpertOutput(
            logits=logits,
            attn_weights=attn_w,
            contributions=contribs
        )

        return expert_out

