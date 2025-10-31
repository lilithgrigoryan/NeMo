# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import warnings
from collections import defaultdict
from itertools import repeat
from pathlib import Path
from typing import Any, Optional, List, Union

import numpy as np
import torch
from lhotse import CutSet
from lightning import LightningModule
from omegaconf import DictConfig
from peft import PeftModel
from torch import Tensor
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
    loss_parallel,
    parallelize_module,
)
from transformers import GenerationConfig

from nemo.collections.asr.models import FastConformerCTCWithAdapterModel
from nemo.collections.asr.parts.utils.rnnt_utils import Hypothesis
from nemo.collections.asr.parts.utils.timestamp_utils import get_forced_aligned_timestamps_with_external_model
from nemo.collections.common.prompts import PromptFormatter
from nemo.collections.common.tokenizers import AutoTokenizer
from nemo.collections.speechlm2.data.salm_dataset import left_collate_vectors
from nemo.collections.speechlm2.models.salm import SALM
from nemo.collections.speechlm2.models.salm_asr_decoder import SALMWithAsrDecoder
from nemo.collections.speechlm2.parts.hf_hub import HFHubMixin
from nemo.collections.speechlm2.parts.lora import maybe_install_lora
from nemo.collections.speechlm2.parts.optim_setup import configure_optimizers, is_frozen
from nemo.collections.speechlm2.parts.pretrained import load_pretrained_hf, move_embedding, setup_speech_encoder
from nemo.core.neural_types import AudioSignal, LabelsType, LengthsType, MaskType, NeuralType
from nemo.utils import logging




class SALMWithTimestamping(SALMWithAsrDecoder):
    """
    SALM model with timestamping capabilities using an external CTC model for forced alignment.
    
    This class extends SALM to support word-level, character-level, and segment-level timestamps
    by using a FastConformerCTCWithAdapterModel for forced alignment.
    
    The encoder (and optionally preprocessor) can be shared between SALM and the CTC model 
    to avoid storing duplicate copies in memory.
    
    Args:
        cfg: Configuration dictionary containing SALM configuration.
        ctc_model_path: Path to the FastConformerCTCWithAdapterModel checkpoint (.nemo file).
                       This model should have the same encoder as SALM's perception module.
        share_encoder: If True (default), share SALM's encoder with the CTC model to save memory.
                      If False, load a separate encoder from the CTC checkpoint.
        share_preprocessor: If True (default), also share the preprocessor to save additional memory.
                           Only enable if both models use identical preprocessing configurations.
                           If False, each model keeps its own preprocessor.
    """
    
    def __init__(
        self, 
        cfg, 
        ctc_model_path: Optional[str] = None, 
        share_encoder: bool = True,
        share_preprocessor: bool = False
        ,
    ) -> None:
        super().__init__(cfg)
        self.ctc_model = None
        self.ctc_model_path = ctc_model_path
        self.share_encoder = share_encoder
        self.share_preprocessor = share_preprocessor
        
        # Load the external CTC model if path is provided
        if ctc_model_path is not None:
            self._load_ctc_model(
                ctc_model_path, 
                share_encoder=share_encoder,
                share_preprocessor=share_preprocessor
            )
            
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
            
    
    def _load_ctc_model(
        self, 
        model_path: str, 
        share_encoder: bool = True,
        share_preprocessor: bool = True
    ) -> None:
        """
        Load the FastConformerCTCWithAdapterModel from a checkpoint.
        
        If share_encoder is True, the encoder from SALM is shared with the CTC model
        to avoid storing two copies in memory. If share_preprocessor is True, the 
        preprocessor is also shared (only do this if they have identical configs).
        
        Args:
            model_path: Path to the .nemo checkpoint file.
            share_encoder: If True, share SALM's encoder with the CTC model.
            share_preprocessor: If True, share SALM's preprocessor with the CTC model.
                               Only enable if both models use identical preprocessing configs.
        """
        logging.info(f"Loading CTC model for timestamping from: {model_path}")
        logging.info(f"Encoder sharing: {'enabled' if share_encoder else 'disabled'}")
        logging.info(f"Preprocessor sharing: {'enabled' if share_preprocessor else 'disabled'}")
        
        try:
            # Load the CTC model
            self.ctc_model = FastConformerCTCWithAdapterModel.restore_from(
                model_path, 
                map_location=self.device
            )
            # ctc_model_weights = self.get_model_weights(self.ctc_model.encoder)
            # salm_model_weights = self.get_model_weights(self.perception.encoder)
            # print(f"CTC model weights: {ctc_model_weights}")
            # print(f"SALM model weights: {salm_model_weights}")
            
            if share_encoder:
                # Replace CTC model's encoder with SALM's encoder to save memory
                logging.info("Replacing CTC model encoder with SALM's shared encoder")
                
                # Delete the CTC model's encoder to free memory
                del self.ctc_model.encoder
                
                # Share the encoder from SALM's perception module
                self.ctc_model.encoder = self.perception.encoder
                
                memory_saved = "~1x encoder size"
                
                if share_preprocessor:
                    # Share the preprocessor only if requested
                    # Note: Only do this if both models use identical preprocessing configs!
                    logging.info("Replacing CTC model preprocessor with SALM's shared preprocessor")
                    del self.ctc_model.preprocessor
                    self.ctc_model.preprocessor = self.perception.preprocessor
                    memory_saved += " + preprocessor"
                else:
                    logging.info("Keeping separate preprocessor (different configs may be used)")
                
                logging.info(f"Successfully shared encoder{' and preprocessor' if share_preprocessor else ''}")
                logging.info(f"Memory savings: {memory_saved}")
            
            self.ctc_model.eval()
            self.ctc_model.freeze()
            logging.info(f"Successfully loaded CTC model from {model_path}")
            
        except Exception as e:
            logging.error(f"Failed to load CTC model from {model_path}: {e}")
            raise
    
    def load_ctc_model(
        self, 
        model_path: str, 
        share_encoder: Optional[bool] = None,
        share_preprocessor: Optional[bool] = None
    ) -> None:
        """
        Public method to load the CTC model after initialization.
        
        Args:
            model_path: Path to the .nemo checkpoint file.
            share_encoder: If provided, overrides the default share_encoder setting.
            share_preprocessor: If provided, overrides the default share_preprocessor setting.
        """
        self.ctc_model_path = model_path
        if share_encoder is None:
            share_encoder = self.share_encoder
        if share_preprocessor is None:
            share_preprocessor = self.share_preprocessor
        self._load_ctc_model(
            model_path, 
            share_encoder=share_encoder,
            share_preprocessor=share_preprocessor
        )
    
    def verify_encoder_sharing(self) -> dict:
        """
        Verify that encoder sharing is working correctly and return memory statistics.
        
        Returns:
            Dictionary containing encoder sharing status and memory information.
        """
        if self.ctc_model is None:
            return {
                "ctc_model_loaded": False,
                "encoder_shared": False,
                "message": "CTC model not loaded"
            }
        
        # Check if encoders are the same object
        encoders_are_shared = self.ctc_model.encoder is self.perception.encoder
        preprocessors_are_shared = self.ctc_model.preprocessor is self.perception.preprocessor
        
        # Count parameters in encoder
        encoder_params = sum(p.numel() for p in self.perception.encoder.parameters())
        
        # Estimate memory saved (approximate, in MB)
        # Assuming float32 (4 bytes per parameter)
        encoder_memory_mb = (encoder_params * 4) / (1024 ** 2)
        
        result = {
            "ctc_model_loaded": True,
            "encoder_shared": encoders_are_shared,
            "preprocessor_shared": preprocessors_are_shared,
            "encoder_parameters": encoder_params,
            "estimated_memory_saved_mb": encoder_memory_mb if encoders_are_shared else 0,
            "message": (
                f"✓ Encoder and preprocessor are shared - saved ~{encoder_memory_mb:.1f}MB" 
                if encoders_are_shared 
                else "✗ Encoder is NOT shared - using separate copies"
            )
        }
        
        logging.info(result["message"])
        return result
    
    @torch.no_grad()
    def generate_with_timestamps(
        self,
        prompts: list[list[dict[str]]] | torch.Tensor,
        audios: torch.Tensor = None,
        audio_lens: torch.Tensor = None,
        generation_config: GenerationConfig = None,
        batch_size: int = 4,
        viterbi_device: Optional[torch.device] = None,
        segment_separators: Optional[Union[str, List[str]]] = ['.', '?', '!', '...'],
        word_separator: Optional[str] = " ",
        supported_punctuation: Optional[Union[set, List[str]]] = {',', '.', '!', '?'},
        timestamp_type: Optional[Union[str, List[str]]] = "all",
        **generation_kwargs,
    ) -> tuple[torch.Tensor, List[Hypothesis]]:
        """
        Generate LLM answers with word-level timestamps using forced alignment with the external CTC model.
        
        This method performs the following steps:
        1. Generate text predictions using SALM's standard generate() method
        2. Decode the generated token IDs to text
        3. Use the external CTC model to perform forced alignment and extract timestamps
        
        Args:
            prompts: Batch of prompts (same format as SALM.generate())
            audios: Optional time-domain audio signal (B, T)
            audio_lens: Optional length of each audio example
            generation_config: Optional HuggingFace GenerationConfig object
            batch_size: Batch size for timestamping alignment (not for generation)
            viterbi_device: Device to use for viterbi decoding in timestamp extraction
            segment_separators: Segment separators for splitting text into segments
            word_separator: Word separator for splitting text into words
            supported_punctuation: Punctuation marks in the vocabulary
            timestamp_type: Type of timestamps to return ("word", "char", "segment", or "all")
            generation_kwargs: Additional keyword arguments for generation
        
        Returns:
            answer_tokens: Generated token IDs (same as SALM.generate())
            hypotheses: List of Hypothesis objects with timestamp information
        
        Raises:
            RuntimeError: If CTC model is not loaded
        """
        if self.ctc_model is None:
            raise RuntimeError(
                "CTC model is not loaded. Please provide ctc_model_path during initialization "
                "or call load_ctc_model() before using generate_with_timestamps()."
            )
        
        # Step 1: Generate answers using standard SALM generation
        answer_tokens = self.generate(
            prompts=prompts,
            audios=audios,
            audio_lens=audio_lens,
            generation_config=generation_config,
            **generation_kwargs,
        )
        
        # Step 2: Decode generated tokens to text
        # Remove prompt tokens from the generated output
        # Note: This assumes the generation includes the prompt in the output
        batch_texts = []
        hypotheses = []
        for i in range(answer_tokens.shape[0]):
            # Decode the generated tokens
            valid_mask=answer_tokens[i] != self.text_pad_id
            text = self.tokenizer.ids_to_text(answer_tokens[i][valid_mask].tolist())
            batch_texts.append(text)
            hypotheses.append(
                Hypothesis(
                    score=0.0, 
                    y_sequence=answer_tokens[i][valid_mask].tolist(), 
                    text=text
                )
            )
        
        
        # Step 4: Prepare audio for timestamping
        # If audio was provided via prompts, we need to extract it
        if audios is None:
            if isinstance(prompts, torch.Tensor):
                raise ValueError(
                    "Cannot extract timestamps without audio. "
                    "Please provide audio via the 'audios' argument or embed it in 'prompts'."
                )
            maybe_audio = _resolve_audios_in_prompt(
                prompts, 
                sampling_rate=self.sampling_rate, 
                device=self.device
            )
            if maybe_audio is None:
                raise ValueError(
                    "Cannot extract timestamps without audio. "
                    "Please provide audio via the 'audios' argument or embed it in 'prompts'."
                )
            audios, audio_lens = maybe_audio
        
        # Step 5: Use external CTC model for forced alignment
        if viterbi_device is None:
            viterbi_device = self.device
        
        # Convert audio tensors to list format expected by get_forced_aligned_timestamps_with_external_model
        audio_list = []
        for i in range(audios.shape[0]):
            audio_length = audio_lens[i].item() if audio_lens is not None else audios.shape[1]
            audio_list.append(audios[i, :audio_length].cpu().numpy())
        
        # Get timestamps using forced alignment
        hypotheses_with_timestamps = get_forced_aligned_timestamps_with_external_model(
            audio=audio_list,
            external_ctc_model=self.ctc_model,
            main_model_predictions=hypotheses,
            batch_size=batch_size,
            viterbi_device=viterbi_device,
            segment_separators=segment_separators,
            word_separator=word_separator,
            supported_punctuation=supported_punctuation,
            timestamp_type=timestamp_type,
            has_hypotheses=False,
        )
        
        return answer_tokens, hypotheses_with_timestamps
    
    @torch.no_grad()
    def get_timestamps_for_text(
        self,
        audio: Union[str, List[str], np.ndarray, torch.Tensor],
        audio_lens: Optional[torch.Tensor] = None,
        texts: List[str] = None,
        batch_size: int = 4,
        viterbi_device: Optional[torch.device] = None,
        segment_separators: Optional[Union[str, List[str]]] = ['.', '?', '!', '...'],
        word_separator: Optional[str] = " ",
        supported_punctuation: Optional[Union[set, List[str]]] = {',', '.', '!', '?'},
        timestamp_type: Optional[Union[str, List[str]]] = "all",
    ) -> List[Hypothesis]:
        """
        Get timestamps for pre-generated text using forced alignment with the external CTC model.
        
        This is useful when you already have the transcription from SALM and just want to add timestamps.
        
        Args:
            audio: Audio data (path, list of paths, numpy array, or torch tensor)
            audio_lens: Optional length of each audio example (if audio is a tensor)
            texts: List of text transcriptions to align with audio
            batch_size: Batch size for alignment
            viterbi_device: Device to use for viterbi decoding
            segment_separators: Segment separators for splitting text
            word_separator: Word separator for splitting text
            supported_punctuation: Punctuation marks in the vocabulary
            timestamp_type: Type of timestamps to return ("word", "char", "segment", or "all")
        
        Returns:
            List of Hypothesis objects with timestamp information
        
        Raises:
            RuntimeError: If CTC model is not loaded
        """
        if self.ctc_model is None:
            raise RuntimeError(
                "CTC model is not loaded. Please provide ctc_model_path during initialization "
                "or call load_ctc_model() before using get_timestamps_for_text()."
            )
        
        # Create Hypothesis objects from texts
        hypotheses = [Hypothesis(text=text, score=0.0) for text in texts]
        
        # Convert audio to proper format if needed
        if isinstance(audio, torch.Tensor):
            audio_list = []
            for i in range(audio.shape[0]):
                audio_length = audio_lens[i].item() if audio_lens is not None else audio.shape[1]
                audio_list.append(audio[i, :audio_length].cpu().numpy())
            audio = audio_list
        
        if viterbi_device is None:
            viterbi_device = self.device
        
        # Get timestamps using forced alignment
        hypotheses_with_timestamps = get_forced_aligned_timestamps_with_external_model(
            audio=audio,
            external_ctc_model=self.ctc_model,
            main_model_predictions=hypotheses,
            batch_size=batch_size,
            viterbi_device=viterbi_device,
            segment_separators=segment_separators,
            word_separator=word_separator,
            supported_punctuation=supported_punctuation,
            timestamp_type=timestamp_type,
            has_hypotheses=False,
        )
        
        return hypotheses_with_timestamps
