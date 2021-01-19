# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os.path as op
from argparse import Namespace

from fairseq.data import Dictionary, encoders
from fairseq.data.audio.speech_to_text_dataset import (
    S2TDataConfig,
    SpeechToTextDataset,
    SpeechToTextDatasetCreator,
    get_features_or_waveform
)
from fairseq.tasks import LegacyFairseqTask, register_task


logger = logging.getLogger(__name__)


@register_task("speech_to_text")
class SpeechToTextTask(LegacyFairseqTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument("data", help="manifest root path")
        parser.add_argument(
            "--config-yaml",
            type=str,
            default="config.yaml",
            help="Configuration YAML filename (under manifest root)",
        )
        parser.add_argument(
            "--max-source-positions",
            default=6000,
            type=int,
            metavar="N",
            help="max number of tokens in the source sequence",
        )
        parser.add_argument(
            "--max-target-positions",
            default=1024,
            type=int,
            metavar="N",
            help="max number of tokens in the target sequence",
        )
        parser.add_argument(
            '--lang-pairs',
            metavar='STR',
            help='comma-separated list of language pairs: en-de,en-fr,de-fr'
        )
        # args for "Adapters for Multilingual Speech Translation"
        parser.add_argument('--subtask', metavar='STR', default="st",
                            help='subtask', choices=["st", "asr", "joint_st_asr"])
        parser.add_argument('--adapter-enc-dim', type=float, metavar='N',
                            default=0.0, help='Adapter dimension in encoder.')
        parser.add_argument('--adapter-enc-type', type=str,
                            choices=[None, 'per_lang', 'shared'], default=None,
                            help='Type of adapters in encoders (None means not used).') 
        parser.add_argument('--adapter-dec-dim', type=float, metavar='N',
                            default=0.0, help='Adapter dimension in decoder.')
        parser.add_argument('--adapter-dec-heads', type=float, metavar='N',
                            default=0.0, help='Adapter heads in parallel adapter.')  
        parser.add_argument('--adapter-dec-type', type=str,
                            choices=[None, 'per_lang', 'shared'], default=None,
                            help='Type of adapters in encoders (None means not used).')
        parser.add_argument('--adapter-dec-mode', type=str,
                            choices=[None, 'serial', 'parallel'], default="serial",
                            help='Mode of adapters in decoders (None means not used).')
        parser.add_argument('--adapter-dec-parallel-to', type=str,
                            choices=[None, 'self_attn', 'layer'], default="layer",
                            help='position of parallel adapters (parallel to which block).')
        parser.add_argument('--adapter-dec-parallel-weight', type=float, default=1.0,
                            help='Weight to combine parallel adapters to the main branch')
        parser.add_argument('--adapter-dec-parallel-learnable', action='store_true',
                            help='Use learnable or fixed weight.')
        parser.add_argument('--homogeneous-batch', action='store_true',
                            help='Use homogeneous batch in training and evaluation.')
        parser.add_argument('--use-mbart', action='store_true',
                            help='Use mbart initialization.')
        parser.add_argument('--update-state-dict-mbart', action='store_true',
                            help='Update mbart initialization for multihead-attention layer.')

    def __init__(self, args, tgt_dict, adapter_keys=None):
        super().__init__(args)
        self.tgt_dict = tgt_dict
        self.adapter_keys = adapter_keys
        self.data_cfg = S2TDataConfig(op.join(args.data, args.config_yaml))

    @classmethod
    def setup_task(cls, args, **kwargs):
        data_cfg = S2TDataConfig(op.join(args.data, args.config_yaml))
        dict_path = op.join(args.data, data_cfg.vocab_filename)
        if not op.isfile(dict_path):
            raise FileNotFoundError(f"Dict not found: {dict_path}")
        tgt_dict = Dictionary.load(dict_path)

        # Add adapter keys
        tgt_langs = sorted([s.split('-')[-1] for s in args.lang_pairs.split(',')])
        if not args.use_mbart: 
            tgt_lang_tags = [
                SpeechToTextDataset.LANG_TAG_TEMPLATE.format(t) for t in set(tgt_langs)
            ]
        else:
            tgt_lang_tags = [
            SpeechToTextDataset.LANG_TAG_MBART_TEMPLATE.format(t, t.upper()) for t in set(tgt_langs)
        ]
            for i, t in enumerate(tgt_lang_tags):
                if t not in tgt_dict:
                    tgt_lang_tags[i] = t.split("_")[0] + "_XX]"
        assert len(tgt_lang_tags) >= 1
        adapter_keys = []
        for t in tgt_lang_tags:
            idx = tgt_dict.index(t)
            logging.info(f'| {t}: {idx}')
            if args.adapter_dec_type == 'per_lang': # use multilingual dict
                assert idx != tgt_dict.unk_index
                adapter_keys.append(str(idx))
            elif args.adapter_dec_type == 'shared': # use bilingual dict
                adapter_keys.append(tgt_lang_tags[0])

        args.adapter_keys = adapter_keys
        if args.adapter_keys and len(tgt_lang_tags) > 1:
            assert args.homogeneous_batch
        logging.info(f'| tgt_lang_tags: {tgt_lang_tags}')
        logging.info(f'| adapter_keys: {args.adapter_keys}')

        logger.info(
            f"dictionary size ({data_cfg.vocab_filename}): " f"{len(tgt_dict):,}"
        )

        if getattr(args, "train_subset", None) is not None:
            if not all(s.startswith("train") for s in args.train_subset.split(",")):
                raise ValueError('Train splits should be named like "train*".')
        return cls(args, tgt_dict, adapter_keys)

    def build_criterion(self, args):
        from fairseq import criterions

        if self.data_cfg.prepend_tgt_lang_tag and args.ignore_prefix_size != 1:
            raise ValueError(
                'Please set "--ignore-prefix-size 1" since '
                "target language ID token is prepended as BOS."
            )
        return criterions.build_criterion(args, self)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        self.datasets[split] = SpeechToTextDatasetCreator.from_tsv(
            self.args.data,
            self.data_cfg,
            split,
            self.tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
            homogeneous_batch=self.args.homogeneous_batch,
            use_mbart=self.args.use_mbart,
            subtask=self.args.subtask,
        )

    @property
    def target_dictionary(self):
        return self.tgt_dict

    @property
    def source_dictionary(self):
        return None

    def max_positions(self):
        return self.args.max_source_positions, self.args.max_target_positions

    def build_model(self, args):
        args.input_feat_per_channel = self.data_cfg.input_feat_per_channel
        args.input_channels = self.data_cfg.input_channels
        return super(SpeechToTextTask, self).build_model(args)

    def build_generator(
        self,
        models,
        args,
        seq_gen_cls=None,
        extra_gen_cls_kwargs=None,
    ):
        if self.data_cfg.prepend_tgt_lang_tag and args.prefix_size != 1:
            raise ValueError(
                'Please set "--prefix-size 1" since '
                "target language ID token is prepended as BOS."
            )
        lang_token_ids = {
            i
            for s, i in self.tgt_dict.indices.items()
            if SpeechToTextDataset.is_lang_tag(s, use_mbart=self.args.use_mbart)
        }
        logging.info(f'| lang_token_ids to strip from output: {lang_token_ids}')
        extra_gen_cls_kwargs = {"symbols_to_strip_from_output": lang_token_ids}
        return super().build_generator(
            models, args, seq_gen_cls=None, extra_gen_cls_kwargs=extra_gen_cls_kwargs
        )

    def build_tokenizer(self, args):
        logger.info(f"pre-tokenizer: {self.data_cfg.pre_tokenizer}")
        return encoders.build_tokenizer(Namespace(**self.data_cfg.pre_tokenizer))

    def build_bpe(self, args):
        logger.info(f"tokenizer: {self.data_cfg.bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.bpe_tokenizer))

    def get_interactive_tokens_and_lengths(self, lines, encode_fn):
        n_frames = [get_features_or_waveform(p).shape[0] for p in lines]
        return lines, n_frames

    def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
        return SpeechToTextDataset(
            "interactive", False, self.data_cfg, src_tokens, src_lengths
        )
