import torch
import torch.nn as nn

from ..feature_head import build_feature_head
from ..models import META_ARCHITECTURES as registry
from .. import transformer as tr


@registry.register('LSTR')
@registry.register('E2E_LSTR')
class LSTR(nn.Module):

    def __init__(self, cfg):
        super(LSTR, self).__init__()

        # Build ages and long feature heads
        self.num_ages_memories = cfg.MODEL.LSTR.NUM_AGES_MEMORIES
        self.ages_enabled = self.num_ages_memories > 0
        if self.ages_enabled:
            self.feature_head_ages = build_feature_head(cfg)
        self.num_long_memories = cfg.MODEL.LSTR.NUM_LONG_MEMORIES
        self.long_enabled = self.num_long_memories > 0
        if self.long_enabled:
            self.feature_head_long = build_feature_head(cfg)

        # Build work feature head
        self.feature_head_work = build_feature_head(cfg)

        self.d_model = self.feature_head_work.d_model
        self.num_heads = cfg.MODEL.LSTR.NUM_HEADS
        self.dim_feedforward = cfg.MODEL.LSTR.DIM_FEEDFORWARD
        self.dropout = cfg.MODEL.LSTR.DROPOUT
        self.num_classes = cfg.DATA.NUM_CLASSES

        self.output_attn_weights = cfg.MODEL.get('OUTPUT_ATTN_WEIGHTS', False)

        # Build position encoding
        self.pos_encoding = tr.PositionalEncoding(self.d_model, self.dropout)

        # Build ages modules
        if self.ages_enabled:
            self.ages_modules = nn.GRU(self.d_model, self.d_model)
            self.ages_linears = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model, self.d_model),
                nn.LayerNorm(self.d_model),
            )
        else:
            self.register_parameter('ages_modules', None)

        # Build long modules
        if self.long_enabled:
            self.long_queries = nn.ModuleList()
            self.long_modules = nn.ModuleList()
            for param in cfg.MODEL.LSTR.LONG_MODULE:
                if param[0] != -1:
                    self.long_queries.append(
                        nn.Embedding(param[0], self.d_model))
                    long_layer = tr.TransformerDecoderLayer(
                        self.d_model, self.num_heads, self.dim_feedforward,
                        self.dropout)
                    self.long_modules.append(
                        tr.TransformerDecoder(
                            long_layer, param[1],
                            tr.layer_norm(self.d_model, param[2])))
                else:
                    self.long_queries.append(None)
                    long_layer = tr.TransformerEncoderLayer(
                        self.d_model, self.num_heads, self.dim_feedforward,
                        self.dropout)
                    self.long_modules.append(
                        tr.TransformerEncoder(
                            long_layer, param[1],
                            tr.layer_norm(self.d_model, param[2])))
        else:
            self.register_parameter('long_queries', None)
            self.register_parameter('long_modules', None)

        # Build work modules
        param = cfg.MODEL.LSTR.WORK_MODULE
        work_layer = tr.TransformerDecoderLayer(self.d_model, self.num_heads,
                                                self.dim_feedforward,
                                                self.dropout)
        self.work_modules = tr.TransformerDecoder(
            work_layer, param[1], tr.layer_norm(self.d_model, param[2]))

        # Build classifier
        self.classifier = nn.Linear(self.d_model, self.num_classes)

        # Place holder for online inference
        self.memory_bank = None

    def online_inference(self,
                            long_visual_inputs,
                            long_motion_inputs,
                            work_visual_inputs,
                            work_motion_inputs,
                            memory_key_padding_mask=None):
        if self.long_enabled:
            if (long_visual_inputs is not None) and (long_motion_inputs
                                                        is not None):
                # Compute fusion input
                fusion_input_long = self.feature_head_long(
                    long_visual_inputs, long_motion_inputs).transpose(0, 1)
                if self.memory_bank is None:
                    self.memory_bank = fusion_input_long
                else:
                    self.memory_bank = torch.cat(
                        (self.memory_bank, fusion_input_long), dim=0)
                if self.memory_bank.shape[0] > self.num_long_memories:
                    self.memory_bank = self.memory_bank[1:]
            else:
                if self.memory_bank is None:
                    raise RuntimeError('No long term memory')
            fusion_input_long = self.memory_bank

            pos = self.pos_encoding.pe[:self.num_long_memories, :]

            if len(self.long_modules) > 0:
                # Compute long queries
                long_queries = [
                    long_query.weight.unsqueeze(1).repeat(
                        1, fusion_input_long.shape[1], 1)
                    if long_query is not None else None
                    for long_query in self.long_queries
                ]

                # Compute long memories
                long_memories, long_attn_weights = self.long_modules[
                    0].online_inference(
                        long_queries[0],
                        fusion_input_long,
                        pos,
                        memory_key_padding_mask=
                        memory_key_padding_mask[:, self.num_ages_memories:])
                for long_query, long_module in zip(long_queries[1:],
                                                    self.long_modules[1:]):
                    if long_query is not None:
                        long_memories, long_attn_weights = long_module(
                            long_query, long_memories)
                    else:
                        long_memories, long_attn_weights = long_module(
                            long_memories)
            else:
                long_memories = fusion_input_long

        if self.long_enabled:
            memory = long_memories
        else:
            raise RuntimeError('ages memory and long memory cannot be None')

        if True:
            # Compute fusion input
            fusion_input_work = self.pos_encoding(
                self.feature_head_work(
                    work_visual_inputs,
                    work_motion_inputs,
                ).transpose(0, 1),
                padding=self.num_ages_memories + self.num_long_memories)

            # Build tgt mask
            tgt_mask = tr.generate_square_subsequent_mask(
                fusion_input_work.shape[0])
            tgt_mask = tgt_mask.to(fusion_input_work.device)

            # Compute output
            output, attn_weights = self.work_modules(
                fusion_input_work,
                memory=memory,
                tgt_mask=tgt_mask,
            )

        # Compute classification score
        score = self.classifier(output)

        return score.transpose(0, 1)

    def forward(self,
                visual_inputs,
                motion_inputs,
                memory_key_padding_mask=None):
        print(visual_inputs.shape)
        if self.ages_enabled:
            # Compute fusion input
            fusion_input_ages = self.feature_head_ages(
                visual_inputs[:, :self.num_ages_memories]
                if visual_inputs is not None else None,
                motion_inputs[:, :self.num_ages_memories]
                if motion_inputs is not None else None,
            ).transpose(0, 1)

            # Compute ages memories
            ages_memories = self.ages_linears(
                self.ages_modules(fusion_input_ages)[1])

        all_long_attn_weights = []
        if self.long_enabled:
            # Compute fusion input
            fusion_input_long = self.pos_encoding(self.feature_head_long(
                visual_inputs[:, self.num_ages_memories:self.num_ages_memories +
                                self.num_long_memories]
                if visual_inputs is not None else None,
                motion_inputs[:, self.num_ages_memories:self.num_ages_memories +
                                self.num_long_memories]
                if motion_inputs is not None else None,
            ).transpose(0, 1),
                                                    padding=self.
                                                    num_ages_memories)

            if len(self.long_modules) > 0:
                # Compute long queries
                long_queries = [
                    long_query.weight.unsqueeze(1).repeat(
                        1, fusion_input_long.shape[1], 1)
                    if long_query is not None else None
                    for long_query in self.long_queries
                ]

                # Compute long memories
                long_memories, attn_weights = self.long_modules[0](
                    long_queries[0],
                    fusion_input_long,
                    memory_key_padding_mask=
                    memory_key_padding_mask[:, self.num_ages_memories:])
                for long_query, long_module in zip(long_queries[1:],
                                                    self.long_modules[1:]):
                    if long_query is not None:
                        long_memories, attn_weights = long_module(
                            long_query, long_memories)
                    else:
                        long_memories, attn_weights = long_module(long_memories)
                    all_long_attn_weights.append(attn_weights)
            else:
                long_memories = fusion_input_long

        if self.ages_enabled and self.long_enabled:
            memory = torch.cat((ages_memories, long_memories))
        elif self.ages_enabled:
            memory = ages_memories
        elif self.long_enabled:
            memory = long_memories
        else:
            raise RuntimeError('ages memory and long memory cannot be None')

        if True:
            # Compute fusion input
            fusion_input_work = self.pos_encoding(
                self.feature_head_work(
                    visual_inputs[:, self.num_ages_memories +
                                    self.num_long_memories:]
                    if visual_inputs is not None else None,
                    motion_inputs[:, self.num_ages_memories +
                                    self.num_long_memories:]
                    if motion_inputs is not None else None,
                ).transpose(0, 1),
                padding=self.num_ages_memories + self.num_long_memories)

            # Build tgt mask
            tgt_mask = tr.generate_square_subsequent_mask(
                fusion_input_work.shape[0])
            tgt_mask = tgt_mask.to(fusion_input_work.device)

            # Compute output
            output, short_attn_weights = self.work_modules(
                fusion_input_work,
                memory=memory,
                tgt_mask=tgt_mask,
            )

        # Compute classification score
        score = self.classifier(output)

        if self.output_attn_weights:
            return score.transpose(0, 1), (all_long_attn_weights,
                                            short_attn_weights)
        else:
            return score.transpose(0, 1)
