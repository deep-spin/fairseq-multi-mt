# Copyright (c) Facebook, Inc. and its affiliates.

import json
import logging
import time
import os
from pathlib import Path
from argparse import Namespace
from collections import defaultdict

import fairseq.checkpoint_utils
import sentencepiece
import torch
from typing import NamedTuple
from dynalab.handler.base_handler import BaseDynaHandler
from dynalab.tasks.flores_small1 import TaskIO
from fairseq.sequence_generator import MultiSourceSequenceGenerator
from fairseq.tasks.translation import TranslationTask
from fairseq.tasks.translation_multi_simple_epoch import TranslationMultiSimpleEpochTask
from fairseq.data import data_utils
from fairseq.data.multilingual.multilingual_utils import augment_dictionary

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Tell Torchserve to let use do the deserialization
os.environ["TS_DECODE_INPUT_REQUEST"] = "false"


def mapping(languages: str) -> dict:
    return dict(
        tuple(pair.split(":"))
        for pair in languages.strip().replace("\n", "").split(",")
    )


ISO2M100 = mapping(
    """
afr:af,amh:am,ara:ar,asm:as,ast:ast,azj:az,bel:be,ben:bn,bos:bs,bul:bg,
cat:ca,ceb:ceb,ces:cs,ckb:ku,cym:cy,dan:da,deu:de,ell:el,eng:en,est:et,
fas:fa,fin:fi,fra:fr,ful:ff,gle:ga,glg:gl,guj:gu,hau:ha,heb:he,hin:hi,
hrv:hr,hun:hu,hye:hy,ibo:ig,ind:id,isl:is,ita:it,jav:jv,jpn:ja,kam:kam,
kan:kn,kat:ka,kaz:kk,kea:kea,khm:km,kir:ky,kor:ko,lao:lo,lav:lv,lin:ln,
lit:lt,ltz:lb,lug:lg,luo:luo,mal:ml,mar:mr,mkd:mk,mlt:mt,mon:mn,mri:mi,
msa:ms,mya:my,nld:nl,nob:no,npi:ne,nso:ns,nya:ny,oci:oc,orm:om,ory:or,
pan:pa,pol:pl,por:pt,pus:ps,ron:ro,rus:ru,slk:sk,slv:sl,sna:sn,snd:sd,
som:so,spa:es,srp:sr,swe:sv,swh:sw,tam:ta,tel:te,tgk:tg,tgl:tl,tha:th,
tur:tr,ukr:uk,umb:umb,urd:ur,uzb:uz,vie:vi,wol:wo,xho:xh,yor:yo,zho_simp:zh,
zho_trad:zh,zul:zu
"""
)


def _load_augmented_dictionary(path, language_list, lang_tok_style, langtoks_specs):
    d = TranslationTask.load_dictionary(path)
    augment_dictionary(
        dictionary=d,
        language_list=language_list,
        lang_tok_style=lang_tok_style,
        langtoks_specs=langtoks_specs
    )
    return d


class FakeGenerator:
    """Fake sequence generator, that returns the input."""

    def generate(self, models, sample, prefix_tokens=None):
        src_tokens = sample["net_input"]["src_tokens"]
        return [[{"tokens": tokens[:-1]}] for tokens in src_tokens]


class Handler(BaseDynaHandler):
    """Use Fairseq model for translation.
    To use this handler, download one of the Flores pretrained model:

    615M parameters:
        https://dl.fbaipublicfiles.com/flores101/pretrained_models/flores101_mm100_615M.tar.gz
    175M parameters:
        https://dl.fbaipublicfiles.com/flores101/pretrained_models/flores101_mm100_175M.tar.gz

    and extract the files next to this one.
    Notably there should be a "dict.txt" and a "sentencepiece.bpe.model".
    """

    def initialize(self, context):
        """
        load model and extra files.
        """

        logger.info(
            f"Will initialize with system_properties: {context.system_properties}"
        )
        model_pt_path, model_file_dir, device = self._handler_initialize(context)
        config = json.loads(
            (Path(model_file_dir) / "model_generation.json").read_text()
        )
        # todo: set path to dictionaries correctly, possibly overriding config
        self.device = device

        cfg = Namespace(**config)
        task_cfg = Namespace(**cfg.task)
        gen_cfg = cfg.generation

        # translation_cfg = TranslationConfig()  # why this?
        # self.vocab = TranslationTask.load_dictionary("dict.txt")

        self.spm = sentencepiece.SentencePieceProcessor()
        self.spm.Load("sentencepiece.bpe.model")
        logger.info("Loaded sentencepiece.bpe.model")

        assert not config.get("dummy", False)  # don't wanna deal with this

        # generate.py does something like this:
        shared_dict = _load_augmented_dictionary(
            "dict.txt",
            task_cfg.langs,
            task_cfg.lang_tok_style,
            task_cfg.langtoks_specs
        )
        # we can use self.spm.set_vocabulary(vocab) to limit the vocabulary to
        # the vocabulary in dict.txt, where vocab is a list of subword types
        self.spm.set_vocabulary(shared_dict.symbols)
        self.vocab = shared_dict
        task = TranslationMultiSimpleEpochTask(
            task_cfg,
            [],
            {lang: shared_dict for lang in task_cfg.langs},
            True
        )

        task2_langs = ["en", "id", "jv", "ms", "ta", "tl"]

        # now: problem with model loading: the model config includes some
        # paths that we no longer have, and don't care about (for example,
        # places to load embeddings from). I solved this by manually changing
        # the values of the model config parameters in model.pt.
        # todo: add adapter_path argument with comma-delimited paths to adapter
        # modules.
        # alternately, could just call load_model_ensemble several times
        src_adapter_paths = ["src-{}.pt".format(lang) for lang in task2_langs]
        tgt_adapter_paths = ["tgt-{}.pt".format(lang) for lang in task2_langs]

        self.src_adapters = dict()
        self.tgt_adapters = dict()

        def _load_adapter(path, map_location, half=False):
            adapter_dict = torch.load("src-{}.pt".format(lang), map_location=self.device)
            if half:
                for k, v in adapter_dict.items():
                    adapter_dict[k] = v.half()
            return adapter_dict

        half_precision = self.device != "cpu" and self.device != torch.device("cpu")
        for lang in task2_langs:
            self.src_adapters[lang] = _load_adapter("src-{}.pt".format(lang), self.device, half_precision)
            self.tgt_adapters[lang] = _load_adapter("tgt-{}.pt".format(lang), self.device, half_precision)

        # load one and only *one* of the sets of adapter params with each of
        # the source and checkpoint.
        [self.src_model], cfg = fairseq.checkpoint_utils.load_model_ensemble(
            [model_pt_path], task=task, adapter_path=src_adapter_paths[0]
        )
        [self.tgt_model], cfg = fairseq.checkpoint_utils.load_model_ensemble(
            [model_pt_path], task=task, adapter_path=tgt_adapter_paths[0]
        )
        if half_precision:
            self.src_model.half()
            self.tgt_model.half()

        self._current_pair = task2_langs[0], task2_langs[0]

        # we need to load the model twice because we're doing ensembles of
        # two models. But only twice!
        self.src_model.eval().to(self.device)
        self.tgt_model.eval().to(self.device)
        # I'm not sure if I'm using fp16. I need to make sure I am

        logger.info(f"Loaded model from {model_pt_path} to device {self.device}")
        logger.info(
            f"Will use the following config: {json.dumps(config, indent=4)}"
        )
        '''
        self.seq_gens = dict()
        for i, src_lang in enumerate(task2_langs):
            for j, tgt_lang in enumerate(task2_langs):
                if src_lang != tgt_lang:
                    pair = src_lang, tgt_lang
                    pair_models = [src_models[i]] + [tgt_models[j]]
                    self.seq_gens[pair] = SequenceGenerator(
                        pair_models,
                        tgt_dict=self.vocab,
                        beam_size=gen_cfg.get("beam", 1),
                        max_len_a=gen_cfg.get("max_len_a", 1.3),
                        max_len_b=gen_cfg.get("max_len_b", 5),
                        min_len=gen_cfg.get("min_len", 5),
                    )
        '''
        self.sequence_generator = MultiSourceSequenceGenerator(
            [self.src_model],  # just a default
            tgt_dict=self.vocab,
            beam_size=gen_cfg.get("beam", 1),
            max_len_a=gen_cfg.get("max_len_a", 1.3),
            max_len_b=gen_cfg.get("max_len_b", 5),
            min_len=gen_cfg.get("min_len", 5),
        )

        self.taskIO = TaskIO()
        self.initialized = True

    def _set_adapters(self, src_lang, tgt_lang):
        prev_src, prev_tgt = self._current_pair
        if src_lang != prev_src:
            self.src_model.load_adapter(self.src_adapters[src_lang])
            # self.src_model.load_state_dict(self.src_adapters[src_lang], strict=False)
        if tgt_lang != prev_tgt:
            # self.tgt_model.load_state_dict(self.tgt_adapters[tgt_lang], strict=False)
            self.tgt_model.load_adapter(self.tgt_adapters[tgt_lang])
        self._current_pair = src_lang, tgt_lang

    def _generate_sequence(self, src_lang, tgt_lang, input_data):
        self._set_adapters(src_lang, tgt_lang)
        return self.sequence_generator.generate(
            models=[self.src_model, self.tgt_model],
            sample=input_data,
            prefix_tokens=input_data["prefix_tokens"],
        )

    def lang_token(self, lang: str) -> int:
        """Converts the ISO 639-3 language code to MM100 language codes."""
        simple_lang = ISO2M100[lang]
        token = self.vocab.index(f"__{simple_lang}__")
        assert token != self.vocab.unk(), f"Unknown language '{lang}' ({simple_lang})"
        return token

    def tokenize(self, line: str) -> list:
        # todo: rewrite this to make use of reduced vocabulary
        words = self.spm.EncodeAsPieces(line.strip())
        tokens = [self.vocab.index(word) for word in words]
        return tokens

    def preprocess_one(self, sample) -> dict:
        """
        preprocess data into a format that the model can do inference on
        """
        # TODO: this doesn't seem to produce good results. wrong EOS / BOS ?
        tokens = self.tokenize(sample["sourceText"])
        src_token = self.lang_token(sample["sourceLanguage"])
        tgt_token = self.lang_token(sample["targetLanguage"])
        return {
            "src_tokens": [src_token] + tokens + [self.vocab.eos()],
            "src_length": len(tokens) + 1,
            "tgt_token": tgt_token,
        }
        return sample

    def preprocess(self, samples) -> dict:
        samples = [self.preprocess_one(s) for s in samples]
        prefix_tokens = torch.tensor([[s["tgt_token"]] for s in samples])
        src_lengths = torch.tensor([s["src_length"] for s in samples])
        src_tokens = data_utils.collate_tokens(
            [torch.tensor(s["src_tokens"]) for s in samples],
            self.vocab.pad(),
            self.vocab.eos(),
        )
        return {
            "nsentences": len(samples),
            "ntokens": src_lengths.sum().item(),
            "net_input": {
                "src_tokens": src_tokens.to(self.device),
                "src_lengths": src_lengths.to(self.device),
            },
            "prefix_tokens": prefix_tokens.to(self.device),
        }

    def strip_pad(self, sentence):
        assert sentence.ndim == 1
        return sentence[sentence.ne(self.vocab.pad())]

    @torch.no_grad()
    def inference(self, input_data: dict) -> list:
        src = self.vocab.string(input_data["net_input"]["src_tokens"][0])[2:4]
        tgt = self.vocab.string(input_data["prefix_tokens"])[2:4]
        generated = self._generate_sequence(src, tgt, input_data)
        # `generate` returns a list of samples
        # with several hypothesis per sample
        # and a dict per hypothesis.
        # We also need to strip the language token.
        return [hypos[0]["tokens"][1:] for hypos in generated]

    def postprocess(self, inference_output, samples: list) -> list:
        """
        post process inference output into a response.
        response should be a list of json
        the response format will need to pass the validation in
        ```
        dynalab.tasks.flores_small1.TaskIO().verify_response(response)
        ```
        """
        translations = [
            self.vocab.string(self.strip_pad(sentence), "sentencepiece")
            for sentence in inference_output
        ]
        return [
            # Signing required by dynabench, don't remove.
            self.taskIO.sign_response(
                {"id": sample["uid"], "translatedText": translation},
                sample,
            )
            for translation, sample in zip(translations, samples)
        ]


_service = Handler()


def deserialize(torchserve_data: list) -> list:
    samples = []
    for torchserve_sample in torchserve_data:
        data = torchserve_sample["body"]
        # In case torchserve did the deserialization for us.
        if isinstance(data, dict):
            samples.append(data)
        elif isinstance(data, (bytes, bytearray)):
            lines = data.decode("utf-8").splitlines()
            for i, l in enumerate(lines):
                try:
                    samples.append(json.loads(l))
                except Exception as e:
                    logging.error(f"Couldn't deserialize line {i}: {l}")
                    logging.exception(e)
        else:
            logging.error(f"Unexpected payload: {data}")

    return samples


def handle_mini_batch(service, samples):
    n = len(samples)
    start_time = time.time()
    input_data = service.preprocess(samples)
    logger.info(
        f"Preprocessed a batch of size {n} ({n/(time.time()-start_time):.2f} samples / s)"
    )

    start_time = time.time()
    output = service.inference(input_data)
    logger.info(
        f"Inferred a batch of size {n} ({n/(time.time()-start_time):.2f} samples / s)"
    )

    start_time = time.time()
    json_results = service.postprocess(output, samples)
    logger.info(
        f"Postprocessed a batch of size {n} ({n/(time.time()-start_time):.2f} samples / s)"
    )
    return json_results


def _batches(samples, max_batch_size):
    """
    There must be a more elegant way to split the sequence into batches
    """
    n = len(samples)
    batch_samples = []
    for i, sample in enumerate(samples):
        batch_samples.append(sample)
        if len(batch_samples) < max_batch_size and i + 1 < n:
            continue  # ugh

        yield batch_samples
        batch_samples = []


def handle(torchserve_data, context):
    if not _service.initialized:
        _service.initialize(context)
    if torchserve_data is None:
        return None

    start_time = time.time()
    samples = deserialize(torchserve_data)
    # frustrating thing is that I don't know how big these deserialized blocks
    # are, or which language pairs I'm using
    n = len(samples)
    logger.info(
        f"Deserialized a batch of size {n} ({n/(time.time()-start_time):.2f} samples / s)"
    )
    # Adapt this to your model. The GPU has 16Gb of RAM.

    # group samples by language pair
    samples_by_pair = defaultdict(list)
    for i, sample in enumerate(samples):
        pair = sample["sourceLanguage"], sample["targetLanguage"]
        samples_by_pair[pair].append((i, sample))

    # batch separately within each group
    results = [None for i in range(len(samples))]
    max_batch_size = 8
    for pair, pair_samples in samples_by_pair.items():
        for batch in _batches(pair_samples, max_batch_size):
            batch_ix, batch_samples = zip(*batch)
            batch_results = handle_mini_batch(_service, batch_samples)
            for i, br in zip(batch_ix, batch_results):
                results[i] = br

    assert len(results)
    start_time = time.time()
    response = "\n".join(json.dumps(r, indent=None, ensure_ascii=False) for r in results)
    logger.info(
        f"Serialized a batch of size {n} ({n/(time.time()-start_time):.2f} samples / s)"
    )
    return [response]


def _load_test_data(path):
    import pickle
    with open(path, "rb") as f:
        unpickled = pickle.load(f, encoding="utf-8")
        test_data = [json.dumps(d).encode("utf-8") for d in unpickled]
        return test_data


def local_test():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("test_pickle", default="test_data.pickle")
    opt = parser.parse_args()

    test_data = _load_test_data(opt.test_pickle)
    bin_data = b"\n".join(test_data)
    torchserve_data = [{"body": bin_data}]

    manifest = {"model": {"serializedFile": "model.pt"}}
    system_properties = {"model_dir": ".", "gpu_id": 0}

    class Context(NamedTuple):
        system_properties: dict
        manifest: dict

    ctx = Context(system_properties, manifest)
    batch_responses = handle(torchserve_data, ctx)
    split_batch_responses = batch_responses[0].split("\n")

    for i, test_ex in enumerate(test_data):
        single_response = handle([{"body": test_ex}], ctx)[0]
        if split_batch_responses[i] != single_response:
            print("mismatch on example {}".format(i))
            print(test_ex)
            print("batch response:")
            print(split_batch_responses[i])
            print("single response:")
            print(single_response)
            assert split_batch_responses[i] == single_response

    '''
    single_responses = [
        handle([{"body": test_ex}], ctx)[0]
        for test_ex in test_data
    ]
    assert batch_responses == ["\n".join(single_responses)]
    '''


if __name__ == "__main__":
    local_test()
