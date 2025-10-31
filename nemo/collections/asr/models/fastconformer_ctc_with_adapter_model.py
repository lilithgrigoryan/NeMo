from typing import Optional, Tuple

import torch
from omegaconf import DictConfig, OmegaConf, open_dict

from nemo.collections.asr.models.asr_model import ASRModel
from nemo.collections.asr.models.ctc_bpe_models import EncDecCTCModelBPE
from nemo.collections.asr.parts.submodules.conformer_modules import ConformerLayer
from nemo.core.classes.common import typecheck
from nemo.core.neural_types import LengthsType, LogprobsType, LabelsType, NeuralType
from nemo.utils import logging, model_utils

from pathlib import Path

import torch.nn as nn

class ModalityAdapterLinear(nn.Module):
    """Simple linear bottleneck adapter with residual connection."""
    def __init__(self, in_dim, hidden_dim, num_layers, out_dim, dropout=0.1):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.LayerNorm(in_dim if i == 0 else hidden_dim))
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)
        self.proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, lengths=None):
        """
        Args:
            x: [B, C, T] - encoder output
            lengths: [B] - sequence lengths (optional)
        Returns:
            x: [B, C, T] - adapted encoder output
        """
        # Convert from [B, C, T] to [B, T, C] for linear layers
        x = x.transpose(1, 2)  # [B, T, C]
        
        # Apply MLP layers
        x = self.mlp(x)
        x = self.proj(x)
        
        # Convert back to [B, C, T]
        x = x.transpose(1, 2)  # [B, C, T]
        
        return x


class ModalityAdapterConformer(nn.Module):
    """FastConformer-based adapter with 2 layers (uses smaller conv kernel like FastConformer)."""
    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        n_heads: int = 4,
        d_ff: int = None,
        conv_kernel_size: int = 9,  # FastConformer uses smaller kernel (9 vs 31)
        dropout: float = 0.1,
        dropout_att: float = 0.0,
    ):
        super().__init__()
        if d_ff is None:
            d_ff = d_model * 4
        
        self.d_model = d_model
        self.n_layers = n_layers
        
        # Stack of FastConformer-style layers
        self.layers = nn.ModuleList([
            ConformerLayer(
                d_model=d_model,
                d_ff=d_ff,
                n_heads=n_heads,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
                dropout_att=dropout_att,
                self_attention_model='abs_pos',  # Use absolute positional encoding for simplicity
            )
            for _ in range(n_layers)
        ])
        
        self.norm_out = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [B, C, T] - encoder output
            lengths: [B] - sequence lengths (optional)
        Returns:
            x: [B, C, T] - adapted encoder output
        """
        # Store residual for skip connection
        residual = x
        
        # Convert from [B, C, T] to [B, T, C] for Conformer layers
        x = x.transpose(1, 2)  # [B, T, C]
        
        # Create padding mask if lengths are provided
        pad_mask = None
        if lengths is not None:
            batch_size, max_len, _ = x.shape
            pad_mask = torch.arange(max_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
        
        # Apply Conformer layers
        for layer in self.layers:
            x = layer(x, pad_mask=pad_mask)
            if isinstance(x, tuple):  # Handle cache returns
                x = x[0]
        
        # Final normalization
        x = self.norm_out(x)
        
        # Convert back to [B, C, T]
        x = x.transpose(1, 2)  # [B, C, T]
        
        # Residual connection
        return residual + x


class FastConformerCTCWithAdapterModel(EncDecCTCModelBPE):
    """CTC model that supports encoder initialization from other models (ASR/SSL)
    and works with adapter-based finetuning.

    Compatible with configs like `examples/asr/conf/fastconformer/ctc_with_adapter/fastconformer_ctc_with_adapter.yaml`.
    """

    def __init__(self, cfg: DictConfig, trainer=None):
        # Convert to Hydra 1.0 compatible DictConfig (standard NeMo pattern)
        cfg = model_utils.convert_model_config_to_dict_config(cfg)
        cfg = model_utils.maybe_update_config_version(cfg)
        
        super().__init__(cfg=cfg, trainer=trainer)

        # Optional: initialize encoder weights from an external model
        self._maybe_init_from_external_encoder()
        # Build an explicit modality adapter that sits AFTER the frozen encoder
        self._build_modality_adapter()

    def get_model_weights(self, model):
        avg=0
        count=0
        minim=10000000
        maxim=0
        for p in model.named_parameters():
            avg += p[1].mean().item()
            count += 1
            minim = min(minim, p[1].min().item())
            maxim = max(maxim, p[1].max().item())
        avg = avg / count
        return avg, minim, maxim

    @property
    def output_types(self) -> Optional[dict[str, NeuralType]]:  # type: ignore[override]
        return {
            "outputs": NeuralType(('B', 'T', 'D'), LogprobsType()),
            "encoded_lengths": NeuralType(tuple('B'), LengthsType()),
            "greedy_predictions": NeuralType(('B', 'T'), LabelsType()),
        }

    @typecheck()
    def forward(
        self,
        input_signal=None,
        input_signal_length=None,
        processed_signal=None,
        processed_signal_length=None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for p in self.encoder.parameters():
            assert p.requires_grad == False
        
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None
        if (has_input_signal ^ has_processed_signal) == False:
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_len`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(
                input_signal=input_signal,
                length=input_signal_length,
            )

        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoder_output = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        encoded = encoder_output[0]
        encoded_len = encoder_output[1]

        if hasattr(self, 'modality_adapter') and self.modality_adapter is not None:
            encoded = self.modality_adapter(encoded, lengths=encoded_len)

        log_probs = self.decoder(encoder_output=encoded)
        greedy_predictions = log_probs.argmax(dim=-1, keepdim=False)

        return (
            log_probs,
            encoded_len,
            greedy_predictions,
        )

    def _maybe_init_from_external_encoder(self) -> None:
        """If `model.init_from` exists in cfg, copy encoder weights from an ASR model and optionally freeze encoder."""
        init_from = self.cfg.get('init_from', None)
        if not init_from:
            return

        pretrained_model = init_from.get('pretrained_model', None)
        nemo_model = init_from.get('nemo_model', None)
        freeze_encoder = init_from.get('freeze_encoder', True)
        salm_ckpt = init_from.get('salm_ckpt', None)

        if not pretrained_model and not nemo_model and not salm_ckpt:
            return

        # Load ASR model
        try:
            if pretrained_model:
                if pretrained_model.endswith('.nemo'):
                    src_model = ASRModel.restore_from(nemo_model, map_location='cpu')
                else: 
                    src_model = ASRModel.from_pretrained(pretrained_model, map_location='cpu')
                    
                missing, unexpected = self.encoder.load_state_dict(src_model.encoder.state_dict(), strict=False)
                if missing or unexpected:
                    logging.info(
                        f"Loaded external encoder from ASR model with non-strict match. Missing keys: {len(missing)}, Unexpected: {len(unexpected)}"
                    )
            elif salm_ckpt:
                init_from_path = Path(salm_ckpt)
                assert init_from_path.is_dir(), "init_from_path must be a directory containing HF checkpoint"
                logging.warning(f"Loading pretrained weights from {str(init_from_path)}")
                from safetensors import safe_open

                tensors = {}
                with safe_open(init_from_path / "model.safetensors", framework="pt") as f:
                    for k in f.keys():
                        # Only keep encoder weights
                        if "encoder." in k:
                            # in salm models encoder is in perception.encoder.
                            # we need to replace it with encoder.
                            new_k = k.replace("perception.encoder.", "encoder.")
                            tensors[new_k] = f.get_tensor(k)
                missing_keys, unexpected_keys = self.load_state_dict(tensors, strict=False)
                logging.warning(f"Loaded external encoder from SALM checkpoint with non-strict match. Missing keys: {len(missing_keys)}, Unexpected: {len(unexpected_keys)}")
                logging.warning(f"Loaded {len(tensors)} encoder weights")
                logging.warning(f"Loaded encoder weights: {tensors.keys()}")
                logging.warning(f"Missing keys: {missing_keys}")
                logging.warning(f"Unexpected keys: {unexpected_keys}")
                
        except Exception as e:
            logging.warning(f"Failed to load external ASR model for encoder init: {e}")
            return

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False
            # Ensure BN stats won't change
            for m in self.encoder.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    m.track_running_stats = False
            self.encoder.eval()
            self.encoder.freeze()

    def _build_modality_adapter(self) -> None:
        adapter_cfg = self._cfg.get('adapter', None)
        if adapter_cfg is None:
            self.modality_adapter = None
            # Default: freeze encoder for alignment learning even if no adapter provided
            for p in self.encoder.parameters():
                p.requires_grad = False
            return

        # Get encoder d_model
        d_model = getattr(self.encoder, 'd_model', None)
        if d_model is None:
            # Try to infer from cfg
            try:
                d_model = int(self.cfg.encoder.get('d_model'))
            except Exception:
                raise ValueError("Could not infer encoder d_model for modality adapter")

        # Determine adapter type and build accordingly
        adapter_type = adapter_cfg.get('adapter_type', 'linear')
        
        if adapter_type == 'linear':
            # Linear bottleneck adapter
            linear_cfg = adapter_cfg.get('linear', {})
            hidden_dim = int(linear_cfg.get('hidden_dim', 1024))
            num_layers = int(linear_cfg.get('num_layers', 2))
            dropout = float(linear_cfg.get('dropout', 0.0))
            self.modality_adapter = ModalityAdapterLinear(d_model, hidden_dim, num_layers, d_model, dropout)
            
        elif adapter_type == 'conformer':
            # FastConformer-based adapter (uses smaller conv kernel)
            conformer_cfg = adapter_cfg.get('conformer', {})
            n_layers = int(conformer_cfg.get('n_layers', 2))
            n_heads = int(conformer_cfg.get('n_heads', 4))
            d_ff = conformer_cfg.get('d_ff', None)
            if d_ff is not None:
                d_ff = int(d_ff)
            conv_kernel_size = int(conformer_cfg.get('conv_kernel_size', 9))  # FastConformer default
            dropout = float(conformer_cfg.get('dropout', 0.1))
            dropout_att = float(conformer_cfg.get('dropout_att', 0.0))
            
            self.modality_adapter = ModalityAdapterConformer(
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                d_ff=d_ff,
                conv_kernel_size=conv_kernel_size,
                dropout=dropout,
                dropout_att=dropout_att,
            )
        else:
            raise ValueError(f"Unknown adapter_type: {adapter_type}. Supported types: 'linear', 'conformer'")

        logging.info(f"Built {adapter_type} modality adapter with d_model={d_model}")

        # Ensure encoder is frozen by default for alignment learning
        freeze_flag = self._cfg.get('freeze_encoder', True)
        if freeze_flag:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Optionally allow training decoder head if requested
        if getattr(self._cfg, 'unfreeze_decoder', False):
            for p in self.decoder.parameters():
                p.requires_grad = True

    def save_adapters(self, filepath: str, name: str = None):
        # Save only the explicit modality adapter to keep artifact small
        if not hasattr(self, 'modality_adapter') or self.modality_adapter is None:
            raise AttributeError("No modality adapter to save.")
        torch.save({
            'modality_adapter': self.modality_adapter.state_dict(),
            'meta': {
                'd_model': getattr(self.encoder, 'd_model', None),
            }
        }, filepath)


