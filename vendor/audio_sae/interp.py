"""Interpretability utilities for AudioSAE.

These helpers reproduce the core analyses from the paper:

- reconstruction quality (L0, L2, fraction of variance explained)
- per-clip top activations for a given feature (max-activation ranking)
- mel-spectrogram windows around frames where a feature fires
- speech / sounds / music classification of features from per-dataset frequencies
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass
class ReconstructionMetrics:
    l0: float
    l2: float
    normalized_l2: float
    frac_var_explained: float


def reconstruction_metrics(
    activations: torch.Tensor,
    features: torch.Tensor,
    reconstruction: torch.Tensor,
) -> ReconstructionMetrics:
    """Standard SAE reconstruction quality metrics, averaged over frames.

    All tensors are expected to be (..., d). ``features`` should be the sparse
    top-k output used by :meth:`BatchTopKSAE.decode` (i.e. what you get from
    ``sae.encode(x, use_threshold=True)`` at inference time).

    L2 is the per-frame reconstruction norm. ``normalized_l2`` divides that by
    the per-frame norm of the input, giving a scale-free relative error
    comparable across models with different activation magnitudes.
    ``frac_var_explained`` is 1 - Var(residual) / Var(input).
    """
    acts = activations.reshape(-1, activations.shape[-1]).float()
    recon = reconstruction.reshape(-1, reconstruction.shape[-1]).float()
    feats = features.reshape(-1, features.shape[-1]).float()

    l0 = (feats != 0).float().sum(dim=-1).mean().item()
    residual_norm = torch.linalg.norm(acts - recon, dim=-1)
    acts_norm = torch.linalg.norm(acts, dim=-1)
    l2 = residual_norm.mean().item()
    normalized_l2 = (residual_norm / acts_norm.clamp_min(1e-12)).mean().item()
    total_variance = acts.var(dim=0).sum()
    residual_variance = (acts - recon).var(dim=0).sum()
    frac_var_explained = (1.0 - residual_variance / total_variance).item()
    return ReconstructionMetrics(
        l0=l0,
        l2=l2,
        normalized_l2=normalized_l2,
        frac_var_explained=frac_var_explained,
    )


@torch.no_grad()
def aggregate_max_activations(
    sae,
    backbone,
    clips: Iterable[Tuple[str, torch.Tensor]],
    device: Optional[str] = None,
) -> Tuple[List[str], np.ndarray]:
    """Run backbone + SAE over a sequence of clips and record the per-clip
    maximum activation of every feature.

    ``clips`` yields ``(clip_id, waveform)`` pairs where ``waveform`` is a
    ``(1, T)`` tensor at the backbone's native sample rate. Returns
    ``(clip_ids, max_acts)`` with ``max_acts`` of shape ``(num_clips, dict_size)``.
    This is the input to :func:`top_clips_for_feature`.
    """
    device = device or next(sae.parameters()).device
    clip_ids: List[str] = []
    rows: List[np.ndarray] = []
    for clip_id, wav in clips:
        wav = wav.to(device)
        acts = backbone(wav)
        latents = sae.encode(acts, use_threshold=True)
        rows.append(latents[0].max(dim=0).values.cpu().numpy())
        clip_ids.append(clip_id)
    if not rows:
        return [], np.zeros((0, sae.dict_size), dtype=np.float32)
    return clip_ids, np.stack(rows, axis=0)


def top_clips_for_feature(
    max_acts: np.ndarray,
    feature_idx: int,
    top_k: int = 10,
) -> np.ndarray:
    """Return the indices of the ``top_k`` clips with the highest activation
    of ``feature_idx``, sorted descending. Clips where the feature never fires
    are excluded.
    """
    col = max_acts[:, feature_idx]
    order = np.argsort(-col)
    order = order[col[order] > 0]
    return order[:top_k]


def dead_features(max_acts: np.ndarray) -> np.ndarray:
    """Return the indices of features that never fired across the whole
    aggregate table."""
    return np.where(max_acts.max(axis=0) == 0)[0]


def feature_activation_windows(
    waveform: torch.Tensor,
    latents: torch.Tensor,
    feature_idx: int,
    sample_rate: int,
    frame_rate_hz: float = 50.0,
    window_s: float = 0.5,
    activation_threshold: float = 0.0,
    n_mels: int = 80,
    max_windows: int = 250,
) -> List[np.ndarray]:
    """Return a list of log-mel spectrogram windows centred on frames where
    ``feature_idx`` fires.

    Ported from the paper's ``get_mels`` helper. Each window is weighted by
    the feature's activation magnitude, so stacking them and averaging gives
    the "what does this feature respond to" visualisation used in the paper.

    ``waveform`` is ``(1, T)`` at ``sample_rate`` Hz, ``latents`` is
    ``(num_frames, dict_size)`` — i.e. ``sae.encode(...)`` for this clip
    without the batch dim. ``frame_rate_hz`` is the frame rate of the
    backbone (50 Hz for HuBERT and Whisper at 16 kHz).
    """
    import librosa

    if isinstance(latents, torch.Tensor):
        lat = latents.detach().cpu().numpy()
    else:
        lat = latents
    wav = waveform.squeeze(0).detach().cpu().numpy()

    mask = lat[:, feature_idx] > activation_threshold
    act_frames = np.where(mask)[0]
    act_values = lat[act_frames, feature_idx]
    act_samples = act_frames / frame_rate_hz * sample_rate

    mel = librosa.feature.melspectrogram(y=wav, sr=sample_rate, n_mels=n_mels)
    mel = librosa.power_to_db(mel, ref=np.max)

    hop = 512
    window_frames = int(2 * window_s * sample_rate / hop) + 1
    half = window_frames // 2

    windows: List[np.ndarray] = []
    for sample_idx, value in zip(act_samples, act_values):
        if sample_idx - window_s * sample_rate < 0:
            continue
        if sample_idx + window_s * sample_rate >= wav.shape[-1]:
            continue
        centre = round(sample_idx / hop)
        chunk = mel[:, centre - half : centre - half + window_frames]
        if chunk.shape[1] == window_frames:
            windows.append(value * chunk)
        if len(windows) >= max_windows:
            break
    return windows


@dataclass
class DatasetActivationStats:
    """Per-dataset activation bookkeeping used by the paper's classifier.

    ``num_activated[i]`` counts frames where feature ``i`` fired,
    ``audio_num_activated[i]`` counts clips where it fired in at least one
    frame. ``frame_frequency`` / ``audio_frequency`` are the normalised
    versions used by :func:`classify_feature_types`.
    """

    num_activated: np.ndarray
    audio_num_activated: np.ndarray
    act_sum: np.ndarray
    num_tokens: int
    num_samples: int

    @property
    def frame_frequency(self) -> np.ndarray:
        denom = max(1, self.num_tokens)
        return self.num_activated / denom

    @property
    def audio_frequency(self) -> np.ndarray:
        denom = max(1, self.num_samples)
        return self.audio_num_activated / denom


@torch.no_grad()
def collect_dataset_stats(
    sae,
    backbone,
    clips: Iterable[torch.Tensor],
    activation_threshold: float = 0.0,
    device: Optional[str] = None,
) -> DatasetActivationStats:
    """Run backbone + SAE over a dataset and accumulate per-feature fire stats.

    ``clips`` yields ``(1, T)`` waveforms at the backbone's native sample rate.
    Returns a :class:`DatasetActivationStats` with frame- and audio-level
    counts. Feed the result of ``{dataset_name: collect_dataset_stats(...)}``
    to :func:`classify_features` to get speech/sounds/music labels.
    """
    device = device or next(sae.parameters()).device
    dict_size = int(sae.dict_size)
    num_activated = np.zeros(dict_size, dtype=np.int64)
    audio_num_activated = np.zeros(dict_size, dtype=np.int64)
    act_sum = np.zeros(dict_size, dtype=np.float64)
    num_tokens = 0
    num_samples = 0

    for wav in clips:
        wav = wav.to(device)
        acts = backbone(wav)
        latents = sae.encode(acts, use_threshold=True)
        flat = latents.reshape(-1, dict_size)
        fired = flat > activation_threshold
        num_activated += fired.sum(dim=0).cpu().numpy().astype(np.int64)
        act_sum += flat.sum(dim=0).cpu().numpy().astype(np.float64)
        num_tokens += flat.shape[0]
        any_fired = fired.any(dim=0).cpu().numpy().astype(np.int64)
        audio_num_activated += any_fired
        num_samples += 1

    return DatasetActivationStats(
        num_activated=num_activated,
        audio_num_activated=audio_num_activated,
        act_sum=act_sum,
        num_tokens=num_tokens,
        num_samples=num_samples,
    )


def classify_features(
    stats_per_dataset: Mapping[str, "DatasetActivationStats"],
    type_to_dataset: Mapping[str, Sequence[str]],
    frequency_margin: float = 0.04,
    level: str = "frame",
) -> Tuple[np.ndarray, np.ndarray]:
    """Classify features as speech / sounds / music given per-dataset stats.

    Convenience wrapper over :func:`classify_feature_types`: extracts the
    requested frequency array (``level="frame"`` or ``"audio"``) from each
    ``DatasetActivationStats`` and calls the underlying classifier. The paper
    reports both frame-level and audio-level labels.
    """
    if level == "frame":
        freq = {name: s.frame_frequency for name, s in stats_per_dataset.items()}
    elif level == "audio":
        freq = {name: s.audio_frequency for name, s in stats_per_dataset.items()}
    else:
        raise ValueError(f"level must be 'frame' or 'audio', got {level!r}")
    return classify_feature_types(freq, type_to_dataset, frequency_margin=frequency_margin)


CATEGORY_COLORS: Dict[str, Tuple[float, float, float]] = {
    "speech": (1.0, 0.0, 0.0),
    "sounds": (0.0, 0.0, 1.0),
    "music": (0.0, 1.0, 0.0),
    "unassigned": (0.5, 0.5, 0.5),
    "dead": (0.0, 0.0, 0.0),
}
"""Colour palette used for category scatter plots (matches the paper)."""


def plot_feature_tsne(
    sae,
    labels: np.ndarray,
    color_map: Optional[Mapping[str, Tuple[float, float, float]]] = None,
    perplexity: float = 30.0,
    random_state: int = 21,
    ax=None,
):
    """t-SNE scatter of encoder rows, coloured by category label.

    Reproduces the frame-level decomposition plot from section 4 of the
    paper. Features labelled ``"dead"`` are dropped before fitting.
    ``labels`` is the ``(dict_size,)`` output of :func:`classify_features`.
    Returns the matplotlib Axes so the caller can customise it.
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    cmap = color_map if color_map is not None else CATEGORY_COLORS
    weight = sae.encoder.weight.detach().cpu().numpy()
    keep = labels != "dead"
    coords = TSNE(
        n_components=2, perplexity=perplexity, random_state=random_state,
    ).fit_transform(weight[keep])
    kept_labels = labels[keep]

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    for cat, color in cmap.items():
        mask = kept_labels == cat
        if not mask.any():
            continue
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c=[color], alpha=0.6, s=10, label=f"{cat} ({int(mask.sum())})",
        )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(loc="best", frameon=True)
    return ax


FeatureTypeStats = Dict[str, Dict[str, np.ndarray]]


def classify_feature_types(
    frame_freq_per_dataset: Mapping[str, np.ndarray],
    type_to_dataset: Mapping[str, Sequence[str]],
    frequency_margin: float = 0.04,
) -> Tuple[np.ndarray, np.ndarray]:
    """Assign each feature a type based on per-dataset activation frequency.

    Reproduces the speech / sounds / music labelling from cell 17 of the
    paper notebook. For each feature we find the category with the highest
    mean frequency across its datasets; the feature is assigned to that
    category only if its lead over every other category is at least
    ``frequency_margin``. Otherwise it is marked as "unassigned" (or "dead"
    if it never fired anywhere).

    ``frame_freq_per_dataset[dataset_name]`` is a ``(dict_size,)`` array of
    per-frame activation frequencies. ``type_to_dataset`` maps a label like
    ``"speech"`` to the list of dataset names that belong to it.

    Returns ``(labels, margins)``:

    - ``labels``: ``(dict_size,)`` string array of the assigned label
      (``"speech"``, ``"sounds"``, ``"music"``, ``"unassigned"``, ``"dead"``).
    - ``margins``: ``(dict_size,)`` float array of the lead of the chosen
      category over the runner-up; 0 for unassigned and dead features.
    """
    categories = list(type_to_dataset.keys())
    cat_freqs = np.stack(
        [
            np.mean(
                [frame_freq_per_dataset[d] for d in type_to_dataset[c]], axis=0
            )
            for c in categories
        ],
        axis=0,
    )

    dict_size = cat_freqs.shape[1]
    labels = np.full(dict_size, "unassigned", dtype=object)
    margins = np.zeros(dict_size, dtype=np.float32)

    max_idx = cat_freqs.argmax(axis=0)
    max_freq = cat_freqs[max_idx, np.arange(dict_size)]

    sorted_freq = np.sort(cat_freqs, axis=0)
    runner_up = sorted_freq[-2]
    lead = max_freq - runner_up

    assigned = lead >= frequency_margin
    for i, cat in enumerate(categories):
        pick = assigned & (max_idx == i)
        labels[pick] = cat
        margins[pick] = lead[pick]

    labels[max_freq == 0] = "dead"
    return np.asarray(labels, dtype=object), margins


PAPER_TYPE_TO_DATASET: Dict[str, List[str]] = {
    "speech": [
        "ls-test-clean",
        "demand-clean",
        "demand-noisy",
        "iemocap",
        "esd",
        "expresso",
        "meld",
    ],
    "sounds": [
        "wham",
        "fsd50k",
        "vocalsound",
        "nonspeech",
        "esc50",
        "vggsound",
    ],
    "music": ["mtg"],
}
"""Dataset grouping used in the paper's feature-type classification."""


__all__ = [
    "ReconstructionMetrics",
    "DatasetActivationStats",
    "CATEGORY_COLORS",
    "PAPER_TYPE_TO_DATASET",
    "reconstruction_metrics",
    "aggregate_max_activations",
    "top_clips_for_feature",
    "dead_features",
    "feature_activation_windows",
    "collect_dataset_stats",
    "classify_features",
    "classify_feature_types",
    "plot_feature_tsne",
]
