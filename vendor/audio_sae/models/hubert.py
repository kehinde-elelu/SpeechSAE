from typing import Union

import torch
from transformers import HubertModel, Wav2Vec2FeatureExtractor


class MyHubert(torch.nn.Module):
    """Thin wrapper around `transformers.HubertModel` that returns per-layer
    hidden states so a Sparse Autoencoder can be attached to any chosen layer.

    Parameters
    ----------
    checkpoint_path : str
        HuggingFace model id (e.g. facebook/hubert-base-ls960) or a local
        path to a checkpoint directory.
    sae_after_layer : int or str
        Layer index whose hidden state will be fed to the SAE. Pass the string
        "all" to return activations from every transformer layer instead.
    sample_rate : int, optional
        Expected audio sample rate, default 16_000 Hz.
    chunk_length : int, optional
        Max number of samples per forward pass, default 30 seconds.
    local_files_only : bool, optional
        Pass through to HubertModel.from_pretrained. Default False so the
        weights download from HuggingFace on first use.
    """

    def __init__(
        self,
        checkpoint_path: str,
        sae_after_layer: Union[int, str],
        sample_rate: int = 16000,
        chunk_length: int = 16000 * 30,
        local_files_only: bool = False,
    ):
        super().__init__()
        self.sae_after_layer = sae_after_layer - 1  # -1 because we are matching layer_x to index x-1
        self.chunk_length = chunk_length
        self.sample_rate = sample_rate
        self.is_large = "large" in checkpoint_path

        self.feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=sample_rate,
            padding_value=0.0,
            padding_side="right",
            return_attention_mask=self.is_large,
            do_normalize=True,
        )

        self.model = HubertModel.from_pretrained(
            checkpoint_path, local_files_only=local_files_only
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        assert signal.shape[0] == 1, "batch size must be 1"

        if signal.dim() == 3:
            signal = signal.squeeze(1)

        feats = self.feature_extractor(
            signal, sampling_rate=self.sample_rate, return_tensors="pt"
        )["input_values"]
        feats = feats.squeeze(1).to(self.model.device)

        hidden_states = self.model(feats, output_hidden_states=True)["hidden_states"]
        hidden_states = torch.stack(hidden_states)

        if isinstance(self.sae_after_layer, int):
            return hidden_states[self.sae_after_layer]
        return hidden_states
