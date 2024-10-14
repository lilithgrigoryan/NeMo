# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig

from nemo.collections.asr.parts.utils import rnnt_utils
from nemo.collections.asr.parts.utils.asr_confidence_utils import ConfidenceMethodMixin
from nemo.collections.common.parts.optional_cuda_graphs import WithOptionalCudaGraphs


class LoopLabelsState:
    """
    State for Loop Labels algorithm. Used only with CUDA graphs.
    In initialization phase it is possible to assign values (tensors) to the state.
    For algorithm code the storage should be reused (prefer copy data instead of assigning tensors).
    """

    max_time: int  # maximum length of internal storage for time dimension
    batch_size: int  # (maximum) length of internal storage for batch dimension
    device: torch.device  # device to store preallocated tensors

    all_durations: torch.Tensor

    encoder_output_projected: torch.Tensor  # projected output from the encoder for decoding algorithm
    encoder_output_length: torch.Tensor  # length of the (projected) output from the encoder

    labels: torch.Tensor  # storage for current labels
    scores: torch.Tensor  # storage for current scores

    batch_indices: torch.Tensor  # indices of elements in batch (constant, range [0, batch_size-1])

    time_indices: torch.Tensor  # current time indices for each element in batch
    safe_time_indices: torch.Tensor  # current time indices, but guaranteed to be < encoder_output_length
    time_indices_current_labels: torch.Tensor  # time indices for found labels (corresponding to `labels` field)
    last_timesteps: torch.Tensor  # indices of the last timesteps for each element (encoder_output_length - 1)

    active_mask: torch.Tensor  # mask for active hypotheses (the decoding is finished for the utterance if it is False)
    advance_mask: torch.Tensor  # mask for "advancing" hypotheses (blank is found for the element on the current step)
    blank_mask: torch.Tensor  # if the element is blank
    # if the element was active on the previous step: to identify the end of decoding and store final hidden state
    active_mask_prev: torch.Tensor
    became_inactive_mask: torch.Tensor  # mask for elements that became inactive (end of decoding)

    active_mask_any: torch.Tensor  # 0-dim bool tensor, condition for outer loop ('any element is still active')
    advance_mask_any: torch.Tensor  # 0-dim bool tensor, condition for inner loop ('should advance any index')

    last_decoder_state: Any  # last state from the decoder, needed for the output
    decoder_state: Any  # current decoder state
    decoder_output: torch.Tensor  # output from the decoder (projected)

    batched_hyps: rnnt_utils.BatchedHyps  # batched hypotheses - decoding result
    alignments: Optional[rnnt_utils.BatchedAlignments] = None  # batched alignments

    def __init__(
        self,
        batch_size: int,
        max_time: int,
        encoder_dim: int,
        max_symbols: int,
        device: torch.device,
        float_dtype: torch.dtype,
        logits_dim: int,
        preserve_alignments=False,
        preserve_frame_confidence=False,
        include_duration_confidence: bool = False,
    ):
        """

        Args:
            batch_size: batch size for encoder output storage
            max_time: maximum time for encoder output storage
            encoder_dim: last dimension for encoder output storage (projected encoder output)
            max_symbols: max symbols per step (to avoid infinite looping and pre-allocate storage)
            device: device to store tensors
            float_dtype: default float dtype for tensors (should match projected encoder output)
            logits_dim: output dimension for Joint
            preserve_alignments: if alignments are needed
            preserve_frame_confidence: if frame confidence is needed
            include_duration_confidence: if duration confidence is needed to be added to the frame confidence
        """
        self.device = device
        self.float_dtype = float_dtype
        self.batch_size = batch_size
        self.max_time = max_time

        self.encoder_output_projected = torch.zeros(
            (self.batch_size, self.max_time, encoder_dim),
            dtype=float_dtype,
            device=self.device,
        )
        self.encoder_output_length = torch.zeros((self.batch_size,), dtype=torch.long, device=self.device)

        self.labels = torch.zeros([self.batch_size], dtype=torch.long, device=self.device)
        self.scores = torch.zeros([self.batch_size], dtype=float_dtype, device=self.device)

        # indices of elements in batch (constant)
        self.batch_indices = torch.arange(self.batch_size, dtype=torch.long, device=self.device)

        self.time_indices = torch.zeros_like(self.batch_indices)
        self.safe_time_indices = torch.zeros_like(self.batch_indices)
        self.time_indices_current_labels = torch.zeros_like(self.time_indices)
        self.last_timesteps = torch.zeros_like(self.time_indices)

        self.active_mask = torch.zeros([self.batch_size], dtype=torch.bool, device=self.device)
        self.advance_mask = torch.zeros_like(self.active_mask)
        self.blank_mask = torch.zeros_like(self.active_mask)
        self.active_mask_prev = torch.zeros_like(self.active_mask)
        self.became_inactive_mask = torch.zeros_like(self.active_mask)

        self.active_mask_any = torch.tensor(True, device=self.device, dtype=torch.bool)
        self.advance_mask_any = torch.tensor(True, device=self.device, dtype=torch.bool)

        self.batched_hyps = rnnt_utils.BatchedHyps(
            batch_size=self.batch_size,
            init_length=self.max_time * max_symbols,
            device=self.device,
            float_dtype=float_dtype,
        )
        if preserve_alignments or preserve_frame_confidence:
            self.alignments = rnnt_utils.BatchedAlignments(
                batch_size=batch_size,
                logits_dim=logits_dim,
                init_length=max_time * (max_symbols + 1),
                device=self.device,
                float_dtype=self.float_dtype,
                store_alignments=preserve_alignments,
                store_frame_confidence=preserve_frame_confidence,
                with_duration_confidence=include_duration_confidence,
            )
        else:
            self.alignments = None

    def need_reinit(self, encoder_output_projected: torch.Tensor) -> bool:
        """Check if need to reinit state: larger batch_size/max_time, or new device"""
        return (
            self.batch_size < encoder_output_projected.shape[0]
            or self.max_time < encoder_output_projected.shape[1]
            or self.device.index != encoder_output_projected.device.index
        )


@dataclass
class SeparateGraphsLoopLabels:
    """Class to store Cuda graphs for decoding when separate graphs are used"""

    before_outer_loop: torch.cuda.CUDAGraph = field(default_factory=torch.cuda.CUDAGraph)
    before_inner_loop: torch.cuda.CUDAGraph = field(default_factory=torch.cuda.CUDAGraph)
    inner_loop_code: torch.cuda.CUDAGraph = field(default_factory=torch.cuda.CUDAGraph)
    after_inner_loop: torch.cuda.CUDAGraph = field(default_factory=torch.cuda.CUDAGraph)


class BeamBatchedTDTLoopLabelsComputer(WithOptionalCudaGraphs, ConfidenceMethodMixin):
    """
    Label Looping algorithm implementation: optimized batched greedy decoding. Callable.
    Iterates over labels, on each step finding the next non-blank label
    (evaluating Joint multiple times in inner loop); It uses a minimal possible amount of calls
    to prediction network (with maximum possible batch size),
    which makes it especially useful for scaling the prediction network.
    During decoding all active hypotheses ("texts") have the same lengths.
    """

    INITIAL_MAX_TIME = 375  # initial max time, used to init state for Cuda graphs

    separate_graphs: Optional[SeparateGraphsLoopLabels]
    full_graph: Optional[torch.cuda.CUDAGraph]
    state: Optional[LoopLabelsState]

    def __init__(
        self,
        decoder,
        joint,
        beam_size: int,
        blank_index: int,
        durations: Union[list[int], ListConfig[int]],
        max_symbols_per_step: Optional[int] = None,
        preserve_alignments=False,
        preserve_frame_confidence=False,
        include_duration_confidence: bool = False,
        confidence_method_cfg: Optional[DictConfig] = None,
    ):
        """
        Init method.
        Args:
            decoder: Prediction network from RNN-T
            joint: Joint module from RNN-T
            blank_index: index of blank symbol
            durations: list of TDT durations, e.g., [0, 1, 2, 4, 8]
            max_symbols_per_step: max symbols to emit on each step (to avoid infinite looping)
            preserve_alignments: if alignments are needed
            preserve_frame_confidence: if frame confidence is needed
            include_duration_confidence: if duration confidence is needed to be added to the frame confidence
            confidence_method_cfg: config for the confidence
        """
        super().__init__()
        self.decoder = decoder
        self.joint = joint
        # keep durations on CPU to avoid side effects in multi-gpu environment
        self.durations = torch.tensor(list(durations), device="cpu").to(torch.long)
        self._blank_index = blank_index
        self.max_symbols = max_symbols_per_step
        self.preserve_alignments = preserve_alignments
        self.preserve_frame_confidence = preserve_frame_confidence
        self.include_duration_confidence = include_duration_confidence
        self._SOS = self._blank_index
        self._init_confidence_method(confidence_method_cfg=confidence_method_cfg)
        assert self._SOS == self._blank_index  # "blank as pad" algorithm only

        self.state = None
        self.full_graph = None
        self.separate_graphs = None
        
        self.beam_size = beam_size
        
        self.max_steps = 4

    def loop_labels_torch(
        self,
        encoder_output: torch.Tensor,
        encoder_output_length: torch.Tensor,
    ) -> Tuple[rnnt_utils.BatchedHyps, Optional[rnnt_utils.BatchedAlignments], Any]:
        """
        Pure PyTorch implementation

        Args:
            encoder_output: output from the encoder
            encoder_output_length: lengths of the utterances in `encoder_output`
        """
        batch_size, max_time, _unused = encoder_output.shape
        beam_size = self.beam_size
        device = encoder_output.device

        encoder_output = encoder_output.repeat_interleave(self.beam_size, dim=0)
        encoder_output_length = encoder_output_length.repeat_interleave(self.beam_size, dim=0)
        # do not recalculate joint projection, project only once
        encoder_output_projected = self.joint.project_encoder(encoder_output)
        float_dtype = encoder_output_projected.dtype

        # init output structures: BatchedHyps (for results), BatchedAlignments + last decoder state
        # init empty batched hypotheses
        batched_hyps = rnnt_utils.BatchedBeamHyps(
            batch_size=batch_size,
            init_length=max_time * self.max_symbols if self.max_symbols is not None else max_time,
            device=device,
            float_dtype=float_dtype,
            beam_size=self.beam_size
        )
        # sample state, will be replaced further when the decoding for hypothesis is done
        last_decoder_state = self.decoder.initialize_state(encoder_output)

        # durations
        all_durations = self.durations.to(device, non_blocking=True)
        num_durations = all_durations.shape[0]

        # initial state, needed for torch.jit to compile (cannot handle None)
        state = self.decoder.initialize_state(encoder_output)
        # indices of elements in batch (constant)
        batch_indices = torch.arange(batch_size * self.beam_size, dtype=torch.long, device=device)
        # last found labels - initially <SOS> (<blank>) symbol
        labels = torch.full((batch_size * self.beam_size,), fill_value=self._SOS)

        # time indices
        time_indices = torch.zeros((batch_size * self.beam_size, ), device=device, dtype=torch.long)
        safe_time_indices = torch.zeros_like(time_indices, device=device, dtype=torch.long)  # time indices, guaranteed to be < out_len
        time_indices_current_labels = torch.zeros_like(time_indices, device=device, dtype=torch.long)
        last_timesteps = (encoder_output_length - 1)

        # masks for utterances in batch
        active_samples_mask: torch.Tensor = encoder_output_length > 0
        inner_activate_samples_mask = torch.empty_like(active_samples_mask)
        # advance_mask = torch.empty_like(active_samples_mask)

        # # for storing the last state we need to know what elements became "inactive" on this step
        # active_mask_prev = torch.empty_like(active_samples_mask)
        # became_inactive_mask = torch.empty_like(active_samples_mask)

        is_first_label = True
        # loop while there are active utterances
        iter_count = 0
        while active_samples_mask.any() and iter_count <= 10:
            iter_count += 1
            expansion_labels = [torch.empty((batch_size * self.beam_size, self.beam_size), device=device, dtype=torch.long) for _ in range(self.max_steps)]
            expansion_durations = [torch.empty((batch_size * self.beam_size, self.beam_size), device=device, dtype=torch.long) for _ in range(self.max_steps)]
            expansion_logps = [torch.empty((batch_size * self.beam_size, self.beam_size), device=device, dtype=float_dtype) for _ in range(self.max_steps)]
            expansion_blank_durations = [torch.zeros((batch_size * self.beam_size, self.beam_size), device=device, dtype=torch.long) for _ in range(self.max_steps)]
            expansion_blank_logps = [torch.zeros((batch_size * self.beam_size, self.beam_size), device=device, dtype=float_dtype) for _ in range(self.max_steps)]
        
            batched_hyps.print()
            # active_mask_prev.copy_(active_samples_mask, non_blocking=True)
            # stage 1: get decoder (prediction network) output
            decoder_output, state, *_ = self.decoder.predict(
                labels.unsqueeze(1), state, add_sos=False, batch_size=batch_size
            )
            decoder_output = self.joint.project_prednet(decoder_output)  # do not recalculate joint projection

            blank_loop = 0
            while blank_loop < self.max_steps and inner_activate_samples_mask.any():
                assert((safe_time_indices <= last_timesteps).all())
                time_indices_current_labels.copy_(time_indices, non_blocking=True)
                # stage 2: get joint output, iteratively seeking for non-blank labels
                # blank label in `labels` tensor means "end of hypothesis" (for this index)
                logits = self.joint.joint_after_projection(
                    encoder_output_projected[batch_indices, safe_time_indices].unsqueeze(1),
                    decoder_output).squeeze()
                label_logits = logits[:, :-num_durations]
                duration_logits = logits[:, -num_durations:]
                
                # Compute log probabilities for labels and durations
                label_logp = torch.log_softmax(label_logits, dim=-1)                # [BATCH*BEAM, V+1]
                duration_logp = torch.log_softmax(duration_logits, dim=-1)          # [BATCH*BEAM, DURATIONS]
                blank_logps = label_logp[:, self._blank_index]
                
                # non-blank expansions
                # TODO leave topk labels
                combined_logp = label_logp[:, :-1, None] + duration_logp[:, None, :]    # [BATCH*BEAM, V, DURATIONS]
                combined_logp =  combined_logp.view(batch_size*beam_size, -1)           # [BATCH*BEAM, V * DURATIONS]
                
                if is_first_label:
                    # before first decoding step all the hypothesis in a beam are identical
                    # keeping just first hyp in a beam
                    combined_logp = combined_logp[::self.beam_size]                 # [BATCH, V * DURATIONS]
                    
                    # getting first BEAM combined logp label and duration pairs
                    # indices are in flattened [V+1, DURATIONS] arrays
                    flat_logp, flat_idx = combined_logp.topk(beam_size, dim = -1)   # [BATCH, BEAM]
                    logps, flat_idx = flat_logp.view(-1, 1), flat_idx.view(-1, 1)   # [BATCH*BEAM]
                    
                    # restoring durations and labels
                    durations = all_durations[flat_idx % num_durations]             # [BATCH*BEAM]
                    labels = flat_idx // num_durations                              # [BATCH*BEAM]
                    
                    blank_logps = blank_logps[::self.beam_size].flatten()
                    
                    blank_duration_logps, blank_duration_idx = duration_logp[::self.beam_size].topk(beam_size, dim=-1)
                    blank_duration_logps, blank_duration_idx = blank_logps.flatten(), blank_duration_idx.flatten()
                    blank_durations = all_durations[blank_duration_idx]
                    
                    assert((blank_duration_idx <= 4).all())
                    assert((labels <= 1023).all())
                    
                    assert(durations.shape == torch.Size([batch_size*beam_size, 1]))
                    assert(labels.shape == torch.Size([batch_size*beam_size, 1]))
                    assert(logps.shape == torch.Size([batch_size*beam_size, 1]))
                    
                    is_first_label = False
                else:
                    logps, flat_idx = combined_logp.topk(beam_size, dim = -1)           # [BATCH*BEAM, BEAM]
                    
                    # restoring durations and labels
                    durations = all_durations[flat_idx % num_durations]                 # [BATCH*BEAM, BEAM]
                    labels = flat_idx // num_durations                                  # [BATCH*BEAM, BEAM]
                    
                    blank_duration_logps, blank_duration_idx = duration_logp.max(dim=-1)
                    blank_durations = all_durations[blank_duration_idx]
                    
                    assert((flat_idx % num_durations <= 4).all())
                    assert((labels <= 1023).all())
                    
                    assert(durations.shape == torch.Size([batch_size*beam_size, beam_size]))
                    assert(labels.shape == torch.Size([batch_size*beam_size, beam_size]))
                    assert(logps.shape == torch.Size([batch_size*beam_size, beam_size]))
                
                expansion_labels[blank_loop] = labels
                expansion_durations[blank_loop] = durations if blank_loop == 0 else expansion_blank_durations[blank_loop - 1] + durations
                expansion_logps[blank_loop] = logps if blank_loop == 0 else expansion_blank_logps[blank_loop - 1] + logps
                
                expansion_blank_logps[blank_loop] = blank_logps + blank_duration_logps if blank_loop == 0 else expansion_blank_logps[blank_loop-1] + blank_logps + blank_duration_logps
                expansion_blank_durations[blank_loop] = all_durations[blank_duration_idx] if blank_loop == 0 else expansion_blank_durations[blank_loop-1] + blank_durations
                
                time_indices_current_labels += blank_durations
                torch.minimum(time_indices_current_labels, last_timesteps, out=safe_time_indices)
                torch.less(time_indices_current_labels, encoder_output_length, out=inner_activate_samples_mask)

                blank_loop += 1
            
            expansion_logps = torch.cat(expansion_logps, dim=1)
            expansion_durations = torch.cat(expansion_durations, dim=1)
            expansion_labels = torch.cat(expansion_labels, dim=1)
            
            num_expansions = expansion_logps.shape[1]
            _, expansion_idx = expansion_logps.view(batch_size, -1).topk(beam_size, -1)
            beam_idx = expansion_idx // num_expansions
            expansion_idx = expansion_idx % num_expansions
            
            expansion_logps = expansion_logps.view(batch_size, beam_size, -1)
            expansion_durations = expansion_durations.view(batch_size, beam_size, -1)
            expansion_labels = expansion_labels.view(batch_size, beam_size, -1)
            
            assert((beam_idx<=4).all())
            assert((expansion_idx<=16).all())
            assert((beam_idx<=expansion_logps.shape[1]).all())
            assert((expansion_idx<=expansion_logps.shape[2]).all())
            logps = expansion_logps[torch.arange(batch_size, dtype=torch.long, device=device), beam_idx, expansion_idx].flatten()
            durations = expansion_durations[torch.arange(batch_size, dtype=torch.long, device=device), beam_idx, expansion_idx].flatten()
            labels = expansion_labels[torch.arange(batch_size, dtype=torch.long, device=device), beam_idx, expansion_idx].flatten()
            
            print(labels)
            print(durations)
            
            time_indices += durations
            torch.minimum(time_indices, last_timesteps, out=safe_time_indices)
            torch.less(time_indices, encoder_output_length, out=active_samples_mask)
            
            # # stage 3: filter labels and state, store hypotheses
            # # select states for hyps that became inactive (is it necessary?)
            # # this seems to be redundant, but used in the `loop_frames` output
            # torch.ne(active_samples_mask, active_mask_prev, out=became_inactive_mask)
            # self.decoder.batch_replace_states_mask(
            #     src_states=state,
            #     dst_states=last_decoder_state,
            #     mask=became_inactive_mask,
            # )

            # # store hypotheses
            # if self.max_symbols is not None:
            #     # pre-allocated memory, no need for checks
            #     batched_hyps.add_results_masked_no_checks_(
            #         active_samples_mask,
            #         labels,
            #         time_indices_current_labels,
            #         scores,
            #         batch_idx=more_batch_idx
            #     )
            # else:
            #     # auto-adjusted storage
            #     batched_hyps.add_results_masked_(
            #         active_samples_mask,
            #         labels,
            #         time_indices_current_labels,
            #         scores,
            #         batch_idx=more_batch_idx
            #     )

            # # stage 4: to avoid looping, go to next frame after max_symbols emission
            # if self.max_symbols is not None:
            #     # if labels are non-blank (not end-of-utterance), check that last observed timestep with label:
            #     # if it is equal to the current time index, and number of observations is >= max_symbols, force blank
            #     force_blank_mask = torch.logical_and(
            #         active_samples_mask,
            #         torch.logical_and(
            #             torch.logical_and(
            #                 labels != self._blank_index,
            #                 batched_hyps.last_timestep_lasts >= self.max_symbols,
            #             ),
            #             batched_hyps.last_timestep == time_indices,
            #         ),
            #     )
            #     time_indices += force_blank_mask  # emit blank => advance time indices
            #     # update safe_time_indices, non-blocking
            #     torch.minimum(time_indices, last_timesteps, out=safe_time_indices)
            #     # same as: active_mask = time_indices < encoder_output_length
            #     torch.less(time_indices, encoder_output_length, out=active_samples_mask)
        return batched_hyps, None, last_decoder_state

    def _get_label_expansions(self,
                              batched_hyps: rnnt_utils.BatchedBeamHyps,
                              label_logp: torch.Tensor,
                              duration_logp: torch.Tensor,
                              all_durations: torch.Tensor,
                              curr_scores: torch.Tensor,
                              is_first_expansion: bool):
        label_logp, labels = label_logp.topk(self.beam_size, dim=-1)            # [ BATCH * BEAM, BEAM]
        combined_logp = label_logp[:, :, None] + duration_logp[:, None, :]      # [ BATCH * BEAM, BEAM, DURATIONS]
        combined_logp = combined_logp.view(combined_logp.shape[0], -1)          # [ BATCH * BEAM, BEAM*DURATIONS]
        total_logp = curr_scores.unsqueeze(-1) + combined_logp          # [ BATCH * BEAM, BEAM*DURATIONS]
        
        if is_first_expansion:
            total_logp, total_logp_idx = total_logp.topk(self.beam_size, dim=-1)
            total_logp, total_logp_idx = total_logp[::self.beam_size].flatten(), total_logp_idx[::self.beam_size].flatten()
            combined_logp = combined_logp.view(combined_logp.shape[0], -1)
            
            beam_idx = torch.arange(self.beam_size, device=total_logp.device).repeat(batched_hyps.batch_size).unsqueeze(0)
            label_idx = total_logp_idx // all_durations.shape[0]
            duration_idx = total_logp_idx % all_durations.shape[0]
        else:
            total_logp = total_logp.view(batched_hyps.batch_size, -1)               # [ BATCH, BEAM*BEAM*DURATIONS]
            total_logp, total_logp_idx = total_logp.topk(self.beam_size, dim=-1)    # [ BATCH, BEAM]
            
            beam_idx = total_logp_idx // (duration_logp.shape[0] * self.beam_size)
            label_idx = total_logp_idx // all_durations.shape[0] % self.beam_size
            duration_idx = total_logp_idx % duration_logp.shape[0]
        
        beam_idx = beam_idx.flatten()
        label_idx = label_idx.flatten()
        duration_idx = duration_idx.flatten()

        logps = combined_logp[torch.arange(labels.shape[0]), label_idx * all_durations.shape[0] + duration_idx]
        labels = labels[torch.arange(labels.shape[0]), label_idx]
        durations = all_durations[duration_idx]
        
        return labels, durations, beam_idx, logps, is_active
    
    # def _get_label_expansions(self, batched_hyps: rnnt_utils.BatchedBeamHyps, logits: torch.Tensor, all_durations: torch.Tensor, is_first_expansion: bool):
    #     num_durations = all_durations.shape[0]
    #     # Get top-k label and duration scores
    #     scores, labels = logits[:, :-num_durations].topk(self.beam_size)
    #     duration_scores = logits[:, -num_durations:]
        
    #     # Compute log probabilities for labels and durations
    #     label_logp = torch.log_softmax(scores, dim=-1)
    #     duration_logp = torch.log_softmax(duration_scores, dim=-1)

    #     batch_size = int(labels.shape[0] / self.beam_size)
    #     # is_first_expansion = False
    #     if is_first_expansion:
    #         # Compute total scores and reshape
    #         total_scores = (label_logp[:, :, None] + duration_logp[:, None, :]).reshape(scores.shape[0], -1)

    #         # Get top-k total scores and their indices
    #         topk_total_score, topk_total_score_idx = total_scores.topk(self.beam_size, dim=-1)
    #         topk_total_score, topk_total_score_idx = topk_total_score[::self.beam_size].flatten(), topk_total_score_idx[::self.beam_size].flatten()
            
    #         # Extract labels and durations using top-k indices
    #         labels = labels[torch.arange(labels.shape[0]), topk_total_score_idx // all_durations.shape[0]]
    #         durations = all_durations[topk_total_score_idx % all_durations.shape[0]]
    #         batch_idx = torch.arange(self.beam_size).repeat(batch_size)
    #     else:
    #         # Compute total scores and reshape
    #         logp = (label_logp[:, :, None] + duration_logp[:, None, :])         # Batch*Beam x Beam x Durations
    #         total_logp = (logp.view(batch_size*self.beam_size, -1) + batched_hyps.scores.unsqueeze(-1)).view(logp.shape[0], -1)      # Batch*Beam x Beam*Durations
            
    #         logp = logp.reshape(scores.shape[0] // self.beam_size, -1)
    #         total_logp = total_logp.reshape(scores.shape[0] // self.beam_size, -1) # Batch x Beam*Beam*Durations
    #         _, topk_idx = total_logp.topk(self.beam_size, dim=-1)
    #         topk_total_score = torch.gather(logp, index=topk_idx, dim=1)
            
    #         topk_total_score, topk_total_score_idx = topk_total_score.flatten(), topk_idx.flatten()

    #         # Extract labels and durations using top-k indices
    #         batch_idx = topk_total_score_idx // all_durations.shape[0] // self.beam_size
    #         labels = labels[torch.arange(labels.shape[0]), topk_total_score_idx // all_durations.shape[0] % self.beam_size]
    #         durations = all_durations[topk_total_score_idx % all_durations.shape[0]]
        
    #     return labels, durations, batch_idx, topk_total_score


    def _before_outer_loop(self):
        """Clear state and compute initial active mask"""
        self.state.batched_hyps.clear_()
        if self.state.alignments is not None:
            self.state.alignments.clear_()

        # initial state
        self.decoder.batch_replace_states_all(
            src_states=self.decoder.initialize_state(self.state.encoder_output_projected),
            dst_states=self.state.decoder_state,
        )
        # last found labels - initially <SOS> (<blank>) symbol
        self.state.labels.fill_(self._SOS)
        self.state.scores.fill_(0.0)

        # time indices
        self.state.time_indices.fill_(0)
        self.state.safe_time_indices.fill_(0)  # safe time indices: guaranteed to be < encoder_output_length
        self.state.time_indices_current_labels.fill_(0)
        torch.sub(self.state.encoder_output_length, 1, out=self.state.last_timesteps)

        # masks for utterances in batch
        # same as: active_mask = self.encoder_output_length > 0
        torch.greater(self.state.encoder_output_length, 0, out=self.state.active_mask)

        # for storing the last state we need to know what elements became "inactive" on this step
        # same as: self.active_mask_any = active_mask.any()
        torch.any(self.state.active_mask, out=self.state.active_mask_any)

    def _before_inner_loop_get_decoder_output(self):
        """Get decoder output"""
        # stage 1: get decoder (prediction network) output
        decoder_output, new_state, *_ = self.decoder.predict(
            self.state.labels.unsqueeze(1), self.state.decoder_state, add_sos=False, batch_size=self.state.batch_size
        )
        self.decoder.batch_replace_states_all(src_states=new_state, dst_states=self.state.decoder_state)
        decoder_output_projected = self.joint.project_prednet(decoder_output)  # do not recalculate joint projection
        self.state.decoder_output.copy_(decoder_output_projected)

    def _before_inner_loop_get_joint_output(self):
        """Get Joint output after decoder output, prepare inner loop to search for all next non-blank labels"""
        # stage 2: get joint output, iteratively seeking for non-blank labels
        # blank label in `labels` tensor means "end of hypothesis" (for this index)
        self.state.active_mask_prev.copy_(self.state.active_mask, non_blocking=True)
        logits = (
            self.joint.joint_after_projection(
                self.state.encoder_output_projected[self.state.batch_indices, self.state.safe_time_indices].unsqueeze(
                    1
                ),
                self.state.decoder_output,
            )
            .squeeze(1)
            .squeeze(1)
        )
        # same as: scores, labels = logits[:, : -self.state.all_durations.shape[0]].max(-1)
        torch.max(logits[:, : -self.state.all_durations.shape[0]], dim=-1, out=(self.state.scores, self.state.labels))
        jump_durations_indices = logits[:, -self.state.all_durations.shape[0] :].argmax(dim=-1)
        durations = self.state.all_durations[jump_durations_indices]

        # search for non-blank labels using joint, advancing time indices for blank labels
        # checking max_symbols is not needed, since we already forced advancing time indices for such cases
        torch.eq(self.state.labels, self._blank_index, out=self.state.blank_mask)
        # blank_mask = self.labels == self._blank_index
        self.state.time_indices_current_labels.copy_(self.state.time_indices, non_blocking=True)
        # for blank labels force duration >= 1
        durations.masked_fill_(torch.logical_and(durations == 0, self.state.blank_mask), 1)
        if self.state.alignments is not None:
            float_dtype = self.state.float_dtype
            self.state.alignments.add_results_masked_no_checks_(
                active_mask=self.state.active_mask,
                time_indices=self.state.time_indices_current_labels,
                logits=logits if self.preserve_alignments else None,
                labels=self.state.labels if self.preserve_alignments else None,
                confidence=(
                    torch.stack(
                        (
                            self._get_confidence_tensor(
                                F.log_softmax(logits[:, : -self.state.all_durations.shape[0]], dim=-1)
                            ).to(dtype=float_dtype),
                            self._get_confidence_tensor(
                                F.log_softmax(logits[:, -self.state.all_durations.shape[0] :], dim=-1)
                            ).to(dtype=float_dtype),
                        ),
                        dim=-1,
                    )
                    if self.include_duration_confidence
                    else (
                        self._get_confidence_tensor(
                            F.log_softmax(logits[:, : -self.state.all_durations.shape[0]], dim=-1)
                        ).to(dtype=float_dtype)
                        if self.preserve_frame_confidence
                        else None
                    )
                ),
            )

        # advance_mask is a mask for current batch for searching non-blank labels;
        # each element is True if non-blank symbol is not yet found AND we can increase the time index
        self.state.time_indices.add_(durations)
        torch.minimum(self.state.time_indices, self.state.last_timesteps, out=self.state.safe_time_indices)
        torch.less(self.state.time_indices, self.state.encoder_output_length, out=self.state.active_mask)
        torch.logical_and(self.state.active_mask, self.state.blank_mask, out=self.state.advance_mask)

        # inner loop: find next non-blank labels (if exist)
        # same as: self.advance_mask_any = advance_mask.any()
        torch.any(self.state.advance_mask, out=self.state.advance_mask_any)

    def _inner_loop_code(self):
        """Find next non-blank labels - one iteration"""
        # same as: time_indices_current_labels[advance_mask] = time_indices[advance_mask], but non-blocking
        # store current time indices to use further for storing the results
        torch.where(
            self.state.advance_mask,
            self.state.time_indices,
            self.state.time_indices_current_labels,
            out=self.state.time_indices_current_labels,
        )
        logits = (
            self.joint.joint_after_projection(
                self.state.encoder_output_projected[self.state.batch_indices, self.state.safe_time_indices].unsqueeze(
                    1
                ),
                self.state.decoder_output,
            )
            .squeeze(1)
            .squeeze(1)
        )
        # get labels (greedy) and scores from current logits, replace labels/scores with new
        # labels[advance_mask] are blank, and we are looking for non-blank labels
        more_scores, more_labels = logits[:, : -self.state.all_durations.shape[0]].max(-1)
        jump_durations_indices = logits[:, -self.state.all_durations.shape[0] :].argmax(dim=-1)
        durations = self.state.all_durations[jump_durations_indices]
        # same as: labels[advance_mask] = more_labels[advance_mask], but non-blocking
        torch.where(self.state.advance_mask, more_labels, self.state.labels, out=self.state.labels)
        # same as: scores[advance_mask] = more_scores[advance_mask], but non-blocking
        torch.where(self.state.advance_mask, more_scores, self.state.scores, out=self.state.scores)

        if self.state.alignments is not None:
            float_dtype = self.state.float_dtype
            self.state.alignments.add_results_masked_no_checks_(
                active_mask=self.state.advance_mask,
                time_indices=self.state.time_indices_current_labels,
                logits=logits if self.preserve_alignments else None,
                labels=more_labels if self.preserve_alignments else None,
                confidence=(
                    torch.stack(
                        (
                            self._get_confidence_tensor(
                                F.log_softmax(logits[:, : -self.state.all_durations.shape[0]], dim=-1)
                            ).to(dtype=float_dtype),
                            self._get_confidence_tensor(
                                F.log_softmax(logits[:, -self.state.all_durations.shape[0] :], dim=-1)
                            ).to(dtype=float_dtype),
                        ),
                        dim=-1,
                    )
                    if self.include_duration_confidence
                    else (
                        self._get_confidence_tensor(
                            F.log_softmax(logits[:, : -self.state.all_durations.shape[0]], dim=-1)
                        ).to(dtype=float_dtype)
                        if self.preserve_frame_confidence
                        else None
                    )
                ),
            )

        # blank_mask = self.labels == self._blank_index
        torch.eq(self.state.labels, self._blank_index, out=self.state.blank_mask)
        # for blank labels force duration >= 1
        durations.masked_fill_(torch.logical_and(durations == 0, self.state.blank_mask), 1)
        # self.time_indices += self.blank_mask
        torch.where(
            self.state.advance_mask,
            self.state.time_indices + durations,
            self.state.time_indices,
            out=self.state.time_indices,
        )

        torch.minimum(self.state.time_indices, self.state.last_timesteps, out=self.state.safe_time_indices)
        torch.less(self.state.time_indices, self.state.encoder_output_length, out=self.state.active_mask)
        torch.logical_and(self.state.active_mask, self.state.blank_mask, out=self.state.advance_mask)
        torch.any(self.state.advance_mask, out=self.state.advance_mask_any)

    def _after_inner_loop(self):
        """Store hypotheses, state for finished hypotheses, avoid looping"""
        # stage 3: filter labels and state, store hypotheses
        # select states for hyps that became inactive (is it necessary?)
        # this seems to be redundant, but used in the `loop_frames` output
        torch.ne(self.state.active_mask, self.state.active_mask_prev, out=self.state.became_inactive_mask)
        self.decoder.batch_replace_states_mask(
            src_states=self.state.decoder_state,
            dst_states=self.state.last_decoder_state,
            mask=self.state.became_inactive_mask,
        )

        self.state.batched_hyps.add_results_masked_no_checks_(
            self.state.active_mask,
            self.state.labels,
            self.state.time_indices_current_labels,
            self.state.scores,
        )

        # stage 4: to avoid looping, go to next frame after max_symbols emission
        # if labels are non-blank (not end-of-utterance), check that last observed timestep with label:
        # if it is equal to the current time index, and number of observations is >= max_symbols, force blank
        force_blank_mask = torch.logical_and(
            self.state.active_mask,
            torch.logical_and(
                torch.logical_and(
                    self.state.labels != self._blank_index,
                    self.state.batched_hyps.last_timestep_lasts >= self.max_symbols,
                ),
                self.state.batched_hyps.last_timestep == self.state.time_indices,
            ),
        )
        self.state.time_indices.add_(force_blank_mask)  # emit blank => advance time indices
        # update safe_time_indices, non-blocking
        torch.minimum(self.state.time_indices, self.state.last_timesteps, out=self.state.safe_time_indices)
        # same as: active_mask = time_indices < encoder_output_length
        torch.less(self.state.time_indices, self.state.encoder_output_length, out=self.state.active_mask)
        torch.any(self.state.active_mask, out=self.state.active_mask_any)

    def __call__(
        self,
        x: torch.Tensor,
        out_len: torch.Tensor,
    ) -> Tuple[rnnt_utils.BatchedHyps, Optional[rnnt_utils.BatchedAlignments], Any]:
        return self.loop_labels_torch(encoder_output=x, encoder_output_length=out_len)


    def maybe_enable_cuda_graphs(self):
       return
   
    def disable_cuda_graphs(self):
        return