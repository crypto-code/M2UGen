# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the GNU General Public License version 3.

from sentencepiece import SentencePieceProcessor
import sentencepiece.sentencepiece_model_pb2 as model
from logging import getLogger
from typing import List
import os


logger = getLogger()


class Tokenizer:
    def __init__(self, model_path: str, num_aud_tokens: int):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        m = model.ModelProto()
        m.ParseFromString(open(model_path, "rb").read())
        special_tokens = [f'[AUD{i}]' for i in range(num_aud_tokens)]
        for token in special_tokens:
            new_token = model.ModelProto().SentencePiece()
            new_token.piece = token
            new_token.score = 0
            if new_token in m.pieces:
                m.pieces.remove(new_token)
            m.pieces.append(new_token)
        with open(model_path, 'wb') as f:
            f.write(m.SerializeToString())  
        self.sp_model = SentencePieceProcessor(model_file=model_path)
        logger.info(f"Reloaded SentencePiece model from {model_path}")

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id}"
        )
        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode_as_ids(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)
