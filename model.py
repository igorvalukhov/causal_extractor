"""
Модель BiLSTM-CRF для извлечения причинно-следственных связей (BIOES).

Архитектура:
  Embedding → BiLSTM → Linear → CRF (опционально simple Viterbi)

Поддерживает обучение, валидацию и инференс.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import List, Dict, Tuple, Optional
import numpy as np


# ──────────────────────────────────────────────────────────────────
# Теги и их маппинги
# ──────────────────────────────────────────────────────────────────

BIOES_TAGS = [
    "O",
    "B-CAUSE", "I-CAUSE", "E-CAUSE", "S-CAUSE",
    "B-EFFECT", "I-EFFECT", "E-EFFECT", "S-EFFECT",
    "<PAD>",
]

TAG2ID = {tag: i for i, tag in enumerate(BIOES_TAGS)}
ID2TAG = {i: tag for tag, i in TAG2ID.items()}
PAD_TAG_ID = TAG2ID["<PAD>"]


# ──────────────────────────────────────────────────────────────────
# Словарь токенов
# ──────────────────────────────────────────────────────────────────

class Vocabulary:
    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(self):
        self.token2id = {self.PAD: 0, self.UNK: 1}
        self.id2token = {0: self.PAD, 1: self.UNK}

    def build(self, dataset: List[Dict], min_freq: int = 1):
        from collections import Counter
        counter = Counter()
        for sample in dataset:
            counter.update(t.lower() for t in sample["tokens"])
        for token, freq in counter.items():
            if freq >= min_freq and token not in self.token2id:
                idx = len(self.token2id)
                self.token2id[token] = idx
                self.id2token[idx] = token
        print(f"Словарь: {len(self.token2id)} токенов")
        return self

    def encode(self, tokens: List[str]) -> List[int]:
        unk = self.token2id[self.UNK]
        return [self.token2id.get(t.lower(), unk) for t in tokens]

    def __len__(self):
        return len(self.token2id)


# ──────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────

class CausalDataset(Dataset):
    def __init__(self, data: List[Dict], vocab: Vocabulary):
        self.samples = []
        for item in data:
            token_ids = vocab.encode(item["tokens"])
            tag_ids = [TAG2ID.get(t, TAG2ID["O"]) for t in item["tags"]]
            self.samples.append((token_ids, tag_ids))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch: List[Tuple[List[int], List[int]]]):
    """Паддинг батча до одинаковой длины."""
    token_seqs, tag_seqs = zip(*batch)
    lengths = [len(s) for s in token_seqs]
    max_len = max(lengths)

    token_tensor = torch.zeros(len(batch), max_len, dtype=torch.long)
    tag_tensor = torch.full((len(batch), max_len), PAD_TAG_ID, dtype=torch.long)

    for i, (toks, tags) in enumerate(zip(token_seqs, tag_seqs)):
        token_tensor[i, :len(toks)] = torch.tensor(toks)
        tag_tensor[i, :len(tags)] = torch.tensor(tags)

    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    return token_tensor, tag_tensor, lengths_tensor


# ──────────────────────────────────────────────────────────────────
# Простой CRF-слой
# ──────────────────────────────────────────────────────────────────

class CRF(nn.Module):
    """Линейная CRF с алгоритмом Витерби для декодирования."""

    def __init__(self, num_tags: int, pad_tag_id: int):
        super().__init__()
        self.num_tags = num_tags
        self.pad_tag_id = pad_tag_id
        # Матрица переходов [from_tag, to_tag]
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        # Запрет переходов с/в PAD
        self.transitions.data[:, pad_tag_id] = -1e4
        self.transitions.data[pad_tag_id, :] = -1e4

    def _compute_log_partition(self, emissions: torch.Tensor,
                                mask: torch.Tensor) -> torch.Tensor:
        """Forward algorithm."""
        batch_size, seq_len, num_tags = emissions.shape
        # Инициализация: первый токен
        score = emissions[:, 0]  # (B, T)
        for t in range(1, seq_len):
            # (B, T, 1) + (B, 1, T) + (1, T, T) → logsumexp по from_tag
            next_score = score.unsqueeze(2) + self.transitions.unsqueeze(0) + emissions[:, t].unsqueeze(1)
            next_score = torch.logsumexp(next_score, dim=1)  # (B, T)
            score = torch.where(mask[:, t].unsqueeze(1), next_score, score)
        return torch.logsumexp(score, dim=1)  # (B,)

    def _score_sentence(self, emissions: torch.Tensor,
                         tags: torch.Tensor,
                         mask: torch.Tensor) -> torch.Tensor:
        """Считает score реальной последовательности тегов."""
        batch_size, seq_len = tags.shape
        score = emissions[:, 0].gather(1, tags[:, 0].unsqueeze(1)).squeeze(1)
        for t in range(1, seq_len):
            m = mask[:, t]
            trans = self.transitions[tags[:, t - 1], tags[:, t]]
            emit = emissions[:, t].gather(1, tags[:, t].unsqueeze(1)).squeeze(1)
            score += (trans + emit) * m.float()
        return score

    def forward(self, emissions: torch.Tensor,
                tags: torch.Tensor,
                mask: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood loss."""
        log_Z = self._compute_log_partition(emissions, mask)
        score = self._score_sentence(emissions, tags, mask)
        return (log_Z - score).mean()

    def decode(self, emissions: torch.Tensor,
               mask: torch.Tensor) -> List[List[int]]:
        """Viterbi декодирование."""
        batch_size, seq_len, num_tags = emissions.shape
        viterbi = emissions[:, 0]  # (B, T)
        backpointers = []

        for t in range(1, seq_len):
            vit_t = viterbi.unsqueeze(2) + self.transitions.unsqueeze(0)  # (B, T, T)
            best_scores, best_tags = vit_t.max(dim=1)  # (B, T)
            backpointers.append(best_tags)
            viterbi = best_scores + emissions[:, t]

        # Восстановление пути
        best_last = viterbi.argmax(dim=1)  # (B,)
        best_paths = []
        for b in range(batch_size):
            path = [best_last[b].item()]
            for bp in reversed(backpointers):
                path.append(bp[b, path[-1]].item())
            path.reverse()
            # Обрезка по маске
            length = mask[b].sum().item()
            best_paths.append(path[:length])
        return best_paths


# ──────────────────────────────────────────────────────────────────
# Основная модель
# ──────────────────────────────────────────────────────────────────

class BiLSTMCRF(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_tags: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_token_id: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_dim, num_tags)
        self.crf = CRF(num_tags, PAD_TAG_ID)

    def _get_emissions(self, tokens: torch.Tensor,
                       lengths: torch.Tensor) -> torch.Tensor:
        emb = self.dropout(self.embedding(tokens))
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        lstm_out = self.dropout(lstm_out)
        return self.linear(lstm_out)

    def forward(self, tokens: torch.Tensor,
                tags: torch.Tensor,
                lengths: torch.Tensor) -> torch.Tensor:
        """Вычисляет CRF-loss."""
        emissions = self._get_emissions(tokens, lengths)
        mask = (tokens != 0)
        # Маскируем PAD-теги нулями для CRF
        tags_crf = tags.clone()
        tags_crf[tags == PAD_TAG_ID] = 0
        return self.crf(emissions, tags_crf, mask)

    def predict(self, tokens: torch.Tensor,
                lengths: torch.Tensor) -> List[List[int]]:
        """Предсказывает теги для батча."""
        with torch.no_grad():
            emissions = self._get_emissions(tokens, lengths)
            mask = (tokens != 0)
            return self.crf.decode(emissions, mask)
