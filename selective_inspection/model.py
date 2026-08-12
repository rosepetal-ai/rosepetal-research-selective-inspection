"""Weak scout model: a DETR-lite image scorer trained from image-level labels only.

This is a self-contained extraction of the scout used in the paper
"Learning When Not to Inspect: Risk-Calibrated Weak Supervision for Efficient
Industrial Visual Inspection" (method id ``scout_image_bce``).

Architecture (paper §3.1):

* ImageNet-pretrained ResNet-18 backbone (via ``timm``) at 256x256 input,
  multi-scale feature maps at strides 8 / 16 / 32.
* Per-scale 1x1 projections to a common model dim, 2-D sine position
  embeddings, tokens concatenated across scales.
* A 2-layer transformer encoder over the tokens.
* A 2-layer transformer decoder with 100 learnable queries (DETR-style).
* Heads: a per-query class logit, a per-query box MLP (present for
  architectural fidelity; it receives no supervision in the weak recipe),
  and the dedicated **image head** — LayerNorm + MLP over the global-mean-
  pooled encoder memory — whose sigmoid is the image score used everywhere
  downstream (exit gate, conformal layer).

Training signal (paper §3.1): binary cross-entropy of the image head against
the image-level OK/NOK label, plus a background BCE on the per-query class
logits (all queries supervised toward "no object"; this term regularises the
shared encoder/backbone). No boxes or masks are ever consumed.

State-dict compatibility: submodule attribute names (``_backbone_module``,
``_input_projs``, ``encoder``, ``decoder``, ``query_embed``, ``class_head``,
``box_head``, ``image_head``) match the research codebase, so checkpoints
trained there load here (``model.load_state_dict(...)``) and vice versa.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

DEFAULT_IMAGE_SIZE = 256
DEFAULT_FEATURE_DIM = 256
DEFAULT_NUM_QUERIES = 100
DEFAULT_NUM_HEADS = 8
DEFAULT_NUM_ENC_LAYERS = 2
DEFAULT_NUM_DEC_LAYERS = 2


@dataclass
class WeakScoutLogits:
    """Raw outputs of :meth:`WeakScout.forward_logits`.

    * ``class_logits`` — ``(B, Q, 1)`` per-query defect-vs-background logits.
    * ``boxes`` — ``(B, Q, 4)`` normalized ``(cx, cy, w, h)`` in ``[0, 1]``
      (unsupervised in the weak recipe; kept for fidelity).
    * ``image_logits`` — ``(B, 1)`` raw image-level logit from the image head.
      ``sigmoid(image_logits)`` is the image score (higher = more anomalous).
    """

    class_logits: torch.Tensor
    boxes: torch.Tensor
    image_logits: torch.Tensor


def _sine_pos_embed_2d(h: int, w: int, dim: int, device: torch.device) -> torch.Tensor:
    """Sine-cosine 2-D positional embedding of shape ``(H*W, dim)`` (DETR-style)."""
    if dim % 2 != 0:
        raise ValueError(f"_sine_pos_embed_2d requires even dim, got {dim}")
    half = dim // 2
    if half % 2 != 0:
        half += 1
    y_embed = torch.linspace(0.0, 1.0, steps=h, device=device).unsqueeze(1).expand(h, w)
    x_embed = torch.linspace(0.0, 1.0, steps=w, device=device).unsqueeze(0).expand(h, w)
    dim_t = torch.arange(half, device=device, dtype=torch.float32)
    dim_t = 10000.0 ** (2.0 * torch.div(dim_t, 2, rounding_mode="floor") / half)
    pos_y = y_embed.unsqueeze(-1) / dim_t
    pos_x = x_embed.unsqueeze(-1) / dim_t
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos = torch.cat([pos_y, pos_x], dim=-1)
    return pos.reshape(h * w, -1)[:, :dim]


class WeakScout(nn.Module):
    """DETR-lite weak scout. Input: ``(B, 3, S, S)`` float in ``[0, 1]``.

    Parameters (all optional; defaults are the paper recipe):

    * ``backbone`` (str) — timm model name, default ``"resnet18"``.
    * ``pretrained`` (bool) — ImageNet-pretrained backbone, default True.
    * ``image_size`` (int) — square input side, default 256.
    * ``feature_dim`` (int) — transformer model dim, default 256.
    * ``num_queries`` (int) — learnable queries, default 100.
    * ``num_heads`` (int) — attention heads, default 8.
    * ``num_encoder_layers`` / ``num_decoder_layers`` (int) — default 2 / 2.
    * ``multi_scale`` (bool) — strides 8/16/32 vs stride 16 only, default True.
    """

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        super().__init__()
        params = dict(params or {})
        self.params = params
        self.backbone_name: str = str(params.get("backbone", "resnet18"))
        self.pretrained: bool = bool(params.get("pretrained", True))
        self.image_size: int = int(params.get("image_size", DEFAULT_IMAGE_SIZE))
        self.feature_dim: int = int(params.get("feature_dim", DEFAULT_FEATURE_DIM))
        self.num_queries: int = int(params.get("num_queries", DEFAULT_NUM_QUERIES))
        self.num_heads: int = int(params.get("num_heads", DEFAULT_NUM_HEADS))
        self.num_encoder_layers: int = int(params.get("num_encoder_layers", DEFAULT_NUM_ENC_LAYERS))
        self.num_decoder_layers: int = int(params.get("num_decoder_layers", DEFAULT_NUM_DEC_LAYERS))
        self.multi_scale: bool = bool(params.get("multi_scale", True))

        self._build_backbone()
        self._build_transformer()
        self._build_heads()

    # --- build -------------------------------------------------------------

    def _build_backbone(self) -> None:
        try:
            import timm
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("WeakScout requires `timm` (pip install timm)") from exc
        # Backbone names may carry a legacy "timm:" prefix (research configs).
        name = self.backbone_name.split(":", 1)[-1]
        out_indices: tuple[int, ...] = (1, 2, 3) if self.multi_scale else (2,)
        self._backbone_module = timm.create_model(
            name,
            pretrained=self.pretrained,
            in_chans=3,
            features_only=True,
            out_indices=out_indices,
        )
        channels = list(self._backbone_module.feature_info.channels())
        self._input_projs = nn.ModuleList(
            [nn.Conv2d(c, self.feature_dim, kernel_size=1, bias=True) for c in channels]
        )

    def _build_transformer(self) -> None:
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.feature_dim,
            nhead=self.num_heads,
            dim_feedforward=4 * self.feature_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.feature_dim,
            nhead=self.num_heads,
            dim_feedforward=4 * self.feature_dim,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_decoder_layers)
        self.query_embed = nn.Embedding(self.num_queries, self.feature_dim)
        nn.init.normal_(self.query_embed.weight, mean=0.0, std=0.02)

    def _build_heads(self) -> None:
        # Per-query class head (RetinaNet prior init; background-supervised
        # in the weak recipe).
        self.class_head = nn.Linear(self.feature_dim, 1, bias=True)
        prior = 0.01
        nn.init.normal_(self.class_head.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.class_head.bias, float(-math.log((1.0 - prior) / prior)))
        # Box head — kept for architectural fidelity / checkpoint
        # compatibility; unsupervised in the weak recipe.
        self.box_head = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.feature_dim, 4),
        )
        for m in self.box_head:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)
        # Image head: the load-bearing output. Decoupled from the per-query
        # head (whose prior-0.01 bias compresses max-query scores near the
        # prior and destroys image-level discrimination); sees a balanced
        # per-image label distribution and uses bias init 0.
        self.image_head = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.GELU(),
            nn.Linear(self.feature_dim, 1),
        )
        for m in self.image_head:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.zeros_(m.bias)

    # --- forward -----------------------------------------------------------

    def _extract_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone -> per-scale projection + sine pos embed -> ``(B, T, D)`` tokens."""
        feats = self._backbone_module(x)
        if not isinstance(feats, (list, tuple)):
            feats = [feats]
        chunks: list[torch.Tensor] = []
        for i, feat in enumerate(feats):
            proj = self._input_projs[i](feat)
            b, c, h, w = proj.shape
            tokens = proj.flatten(2).transpose(1, 2)  # (B, HW, D)
            pos = _sine_pos_embed_2d(h, w, c, x.device).unsqueeze(0)
            chunks.append(tokens + pos)
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks, dim=1)

    def forward_logits(self, images: torch.Tensor) -> WeakScoutLogits:
        """Full forward pass. ``images``: ``(B, 3, H, W)`` float in ``[0, 1]``."""
        if images.dim() != 4:
            raise ValueError(f"images must be (B, C, H, W); got {tuple(images.shape)}")
        tokens = self._extract_tokens(images)
        memory = self.encoder(tokens)  # (B, T, D)
        b = images.shape[0]
        queries = self.query_embed.weight.unsqueeze(0).expand(b, -1, -1)
        tgt = torch.zeros_like(queries) + queries
        decoded = self.decoder(tgt, memory)  # (B, Q, D)
        class_logits = self.class_head(decoded)  # (B, Q, 1)
        boxes = torch.sigmoid(self.box_head(decoded))  # (B, Q, 4)
        pooled = memory.mean(dim=1)  # (B, D)
        image_logits = self.image_head(pooled)  # (B, 1)
        return WeakScoutLogits(class_logits=class_logits, boxes=boxes, image_logits=image_logits)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Convenience forward: image scores ``(B,)`` in ``[0, 1]`` (sigmoid of the image head)."""
        return torch.sigmoid(self.forward_logits(images).image_logits.squeeze(-1))


def weak_bce_loss(
    logits: WeakScoutLogits,
    image_labels: torch.Tensor,
    *,
    class_weight: float = 1.0,
    image_weight: float = 1.0,
) -> dict[str, torch.Tensor]:
    """The weak-supervision loss (paper §3.1) — image labels are the only supervision.

    * ``loss_image`` — BCE of the image head against ``image_labels``
      (``(B,)`` or ``(B, 1)`` float, 1.0 = NOK, 0.0 = OK). Load-bearing term.
    * ``loss_class`` — BCE of every per-query class logit against 0
      (background). In the research codebase this is the empty-box branch of
      the distillation loss; it regularises the shared encoder/backbone.

    Returns a dict with ``loss`` (total) plus detached per-term scalars.
    """
    image_labels = image_labels.reshape(-1, 1).to(logits.image_logits.dtype)
    loss_class = nn.functional.binary_cross_entropy_with_logits(
        logits.class_logits, torch.zeros_like(logits.class_logits), reduction="mean"
    )
    loss_image = nn.functional.binary_cross_entropy_with_logits(
        logits.image_logits, image_labels, reduction="mean"
    )
    total = float(class_weight) * loss_class + float(image_weight) * loss_image
    return {"loss": total, "loss_class": loss_class.detach(), "loss_image": loss_image.detach()}


def build_model(model_cfg: dict[str, Any]) -> WeakScout:
    """Build a :class:`WeakScout` from a config ``model:`` block."""
    return WeakScout(model_cfg)


def save_checkpoint(path: str, model: WeakScout, *, extra: dict[str, Any] | None = None) -> None:
    """Save ``{model_state, model_params, ...extra}`` (loadable with ``weights_only=True``)."""
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "model_params": {
            "backbone": model.backbone_name,
            "pretrained": model.pretrained,
            "image_size": model.image_size,
            "feature_dim": model.feature_dim,
            "num_queries": model.num_queries,
            "num_heads": model.num_heads,
            "num_encoder_layers": model.num_encoder_layers,
            "num_decoder_layers": model.num_decoder_layers,
            "multi_scale": model.multi_scale,
        },
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: str, device: str | torch.device = "cpu") -> tuple[WeakScout, dict[str, Any]]:
    """Load a checkpoint saved by :func:`save_checkpoint` (or a raw state dict).

    Returns ``(model.eval() on device, payload_metadata)``. Raw state dicts
    (e.g. ``final_state.pt`` from the research codebase) are loaded into a
    default paper-recipe model. Pretrained backbone weights are NOT
    re-downloaded — the checkpoint overwrites every parameter.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "model_state" in payload:
        params = dict(payload.get("model_params") or {})
        state = payload["model_state"]
        meta = {k: v for k, v in payload.items() if k != "model_state"}
    else:  # raw state_dict (research-codebase final_state.pt)
        params, state, meta = {}, payload, {}
    params["pretrained"] = False  # never download weights just to overwrite them
    model = WeakScout(params)
    model.load_state_dict(state)
    model.to(torch.device(device))
    model.eval()
    return model, meta


__all__ = [
    "DEFAULT_IMAGE_SIZE",
    "WeakScout",
    "WeakScoutLogits",
    "build_model",
    "load_checkpoint",
    "save_checkpoint",
    "weak_bce_loss",
]
