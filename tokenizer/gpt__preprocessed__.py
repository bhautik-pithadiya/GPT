# NOTE: This file was automatically generated from:
# /home/ksuser/Bhautik/GPT/tokenizer/gpt.py
# DO NOT CHANGE DIRECTLY! 1715778989.3855264
import regex as re
try:
    (Tokenizer,) = ultraimport('__dir__/base/__init__.py', objects_to_import=('Tokenizer',), recurse=True)
except ultraimport.ResolveImportError as e:
    try:
        (Tokenizer,) = ultraimport('__dir__/base.py', objects_to_import=('Tokenizer',), recurse=True)
    except ultraimport.ResolveImportError as e2:
        raise ultraimport.RewrittenImportError(code_info=('from .base import Tokenizer, get_stats, merge, merge_orignial', '/home/ksuser/Bhautik/GPT/tokenizer/gpt.py', 2, 0), object_to_import='Tokenizer', combine=[e, e2]) from None
try:
    (get_stats,) = ultraimport('__dir__/base/__init__.py', objects_to_import=('get_stats',), recurse=True)
except ultraimport.ResolveImportError as e:
    try:
        (get_stats,) = ultraimport('__dir__/base.py', objects_to_import=('get_stats',), recurse=True)
    except ultraimport.ResolveImportError as e2:
        raise ultraimport.RewrittenImportError(code_info=('from .base import Tokenizer, get_stats, merge, merge_orignial', '/home/ksuser/Bhautik/GPT/tokenizer/gpt.py', 2, 0), object_to_import='get_stats', combine=[e, e2]) from None
try:
    (merge,) = ultraimport('__dir__/base/__init__.py', objects_to_import=('merge',), recurse=True)
except ultraimport.ResolveImportError as e:
    try:
        (merge,) = ultraimport('__dir__/base.py', objects_to_import=('merge',), recurse=True)
    except ultraimport.ResolveImportError as e2:
        raise ultraimport.RewrittenImportError(code_info=('from .base import Tokenizer, get_stats, merge, merge_orignial', '/home/ksuser/Bhautik/GPT/tokenizer/gpt.py', 2, 0), object_to_import='merge', combine=[e, e2]) from None
try:
    (merge_orignial,) = ultraimport('__dir__/base/__init__.py', objects_to_import=('merge_orignial',), recurse=True)
except ultraimport.ResolveImportError as e:
    try:
        (merge_orignial,) = ultraimport('__dir__/base.py', objects_to_import=('merge_orignial',), recurse=True)
    except ultraimport.ResolveImportError as e2:
        raise ultraimport.RewrittenImportError(code_info=('from .base import Tokenizer, get_stats, merge, merge_orignial', '/home/ksuser/Bhautik/GPT/tokenizer/gpt.py', 2, 0), object_to_import='merge_orignial', combine=[e, e2]) from None
import torch
GPT2_SPLIT_PATTERN = "'(?:[sdmt]|ll|ve|re)| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+"
GPT4_SPLIT_PATTERN = "'(?i:[sdmt]|ll|ve|re)|[^\\r\\n\\p{L}\\p{N}]?+\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]++[\\r\\n]*|\\s*[\\r\\n]|\\s+(?!\\S)|\\s+"

class GptTokenizer(Tokenizer):

    def __init__(self, pattern=None):
        """
        - pattern: optional string to override the default (GPT-4 split pattern)
        - special_tokens: str -> int dictionary of special tokens
            example: {'<|endoftext|>': 100257}
        """
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern
        self.compiled_pattern = re.compile(self.pattern)
        self.special_tokens = {}
        self.inverse_special_tokens = {}

    def train(self, text, vocab_size, verbose=False):
        assert vocab_size >= 256
        num_merges = vocab_size - 256
        text_chunks = re.findall(self.compiled_pattern, text)
        ids = [list(ch.encode('utf-8')) for ch in text_chunks]
        merges = {}
        vocab = {idx: bytes([idx]) for idx in range(256)}
        for i in range(num_merges):
            stats = {}
            for chunk_ids in ids:
                get_stats(chunk_ids, stats)
            pair = max(stats, key=stats.get)
            idx = 256 + i
            ids = [merge_orignial(chunk_ids, pair, idx) for chunk_ids in ids]
            merges[pair] = idx
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose:
                print(f'merge {i + 1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences')
        self.merges = merges
        self.vocab = vocab

    def register_special_tokens(self, special_tokens):
        self.special_tokens = special_tokens
        self.inverse_special_tokens = {v: k for (k, v) in special_tokens.items()}

    def decode(self, ids):
        part_bytes = []
        ids_list = ids.tolist()
        for idx in ids_list:
            if idx in self.vocab:
                part_bytes.append(self.vocab[idx])
            elif idx in self.inverse_special_tokens:
                part_bytes.append(self.inverse_special_tokens[idx].encode('utf-8'))
            else:
                raise ValueError(f'invalid token id: {idx}')
        text_bytes = b''.join(part_bytes)
        text = text_bytes.decode('utf-8', errors='replace')
        return text

    def pre_encode(self, text):
        text_chunks = re.findall(self.compiled_pattern, text)
        chunks = [chunk.encode('utf-8') for chunk in text_chunks]
        return chunks

    def encode_ordinary(self, text):
        """Encoding that ignores any special tokens."""
        chunks = self.pre_encode(text)
        int_type = torch.int16 if len(self.merges) <= 2 ** 15 else torch.int32
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ids = [list(chunk_bytes) for chunk_bytes in chunks]
        if len(self.merges) == 0:
            return sum(ids, [])
        merges = sorted(list(self.merges), key=lambda p: self.merges[p])
        merges = torch.tensor(merges, dtype=int_type, device=device)
        for (i, chunk_ids) in enumerate(ids):
            chunk_ids = torch.tensor(chunk_ids, dtype=int_type, device=device)
            while len(chunk_ids) >= 2:
                pairs = torch.stack((chunk_ids[:-1], chunk_ids[1:]), dim=1)
                unique = torch.unique(pairs, dim=0)
                is_present = (merges[:, None] == unique[None]).all(-1).any(-1)
                if not is_present.any():
                    break
                pair_index = is_present.nonzero()[0]
                pair = merges[pair_index]
                idx = pair_index.to(chunk_ids.dtype) + 256
                chunk_ids = merge(chunk_ids, pair, idx)
            ids[i] = chunk_ids.cpu().tolist()
        return sum(ids, [])

    def encode(self, text, allowed_special='all'):
        """
        Unlike encode_ordinary, this function handles special tokens.
        allowed_special: can be "all"|"none"|"none_raise" or a custom set of special tokens
        if none_raise, then an error is raised if any special token is encountered in text
        this is the default tiktoken behavior right now as well
        any other behavior is either annoying, or a major footgun
        """
        special = None
        if allowed_special == 'all':
            special = self.special_tokens
        elif allowed_special == 'none':
            special = {}
        elif allowed_special == 'none_raise':
            special = {}
            assert all((token not in text for token in self.special_tokens))
        elif isinstance(allowed_special, set):
            special = {k: v for (k, v) in self.special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f'allowed_special={allowed_special} not understood')
        if not special:
            return self.encode_ordinary(text)
        special_pattern = '(' + '|'.join((re.escape(k) for k in special)) + ')'
        special_chunks = re.split(special_pattern, text)
        ids = []
        for part in special_chunks:
            if part in special:
                ids.append(special[part])
            else:
                ids.extend(self.encode_ordinary(part))
        return ids