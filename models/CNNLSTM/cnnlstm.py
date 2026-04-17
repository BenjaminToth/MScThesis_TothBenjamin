from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class CNNLSTM(nn.Module):
    """CNN-LSTM for EEG artifact classification.

    Architecture:
        1) Temporal CNN feature extractor
        2) LSTM over downsampled time steps
        3) Linear classifier

    Accepted input shapes:
        - (B, C, T)
        - (B, T, C)
        - (B, 1, C, T)
        - (B, 1, T, C)

    Where:
        B = batch size
        C = number of EEG channels
        T = number of time samples

    Output:
        - logits with shape (B, n_classes)
    """

    def __init__(
        self,
        n_channels: int = 17,
        n_samples: Optional[int] = None,
        n_classes: int = 6,
        cnn_hidden_1: int = 64,
        cnn_hidden_2: int = 128,
        kernel_size_1: int = 15,
        kernel_size_2: int = 7,
        pool_size_1: int = 2,
        pool_size_2: int = 2,
        lstm_hidden_size: int = 192,
        lstm_num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.35,
    ) -> None:
        super().__init__()

        if n_channels <= 0:
            raise ValueError("n_channels must be > 0")
        if n_samples is not None and n_samples <= 0:
            raise ValueError("n_samples must be > 0 when provided")
        if n_classes <= 0:
            raise ValueError("n_classes must be > 0")
        if cnn_hidden_1 <= 0 or cnn_hidden_2 <= 0:
            raise ValueError("cnn_hidden_1 and cnn_hidden_2 must be > 0")
        if kernel_size_1 <= 0 or kernel_size_2 <= 0:
            raise ValueError("kernel sizes must be > 0")
        if pool_size_1 <= 0 or pool_size_2 <= 0:
            raise ValueError("pool sizes must be > 0")
        if lstm_hidden_size <= 0:
            raise ValueError("lstm_hidden_size must be > 0")
        if lstm_num_layers <= 0:
            raise ValueError("lstm_num_layers must be > 0")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must satisfy 0.0 <= dropout < 1.0")

        self.n_channels = int(n_channels)
        self.n_samples = int(n_samples) if n_samples is not None else None
        self.n_classes = int(n_classes)
        self.bidirectional = bool(bidirectional)
        self.lstm_hidden_size = int(lstm_hidden_size)

        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=n_channels,
                out_channels=cnn_hidden_1,
                kernel_size=kernel_size_1,
                padding=kernel_size_1 // 2,
                bias=False,
            ),
            nn.BatchNorm1d(cnn_hidden_1),
            nn.ELU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_size_1, stride=pool_size_1),
            nn.Dropout(p=dropout),

            nn.Conv1d(
                in_channels=cnn_hidden_1,
                out_channels=cnn_hidden_2,
                kernel_size=kernel_size_2,
                padding=kernel_size_2 // 2,
                bias=False,
            ),
            nn.BatchNorm1d(cnn_hidden_2),
            nn.ELU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_size_2, stride=pool_size_2),
            nn.Dropout(p=dropout),
        )

        self.lstm = nn.LSTM(
            input_size=cnn_hidden_2,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        lstm_output_size = lstm_hidden_size * (2 if bidirectional else 1)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(lstm_output_size, n_classes),
        )

    def _normalize_input_shape(self, x: torch.Tensor) -> torch.Tensor:
        """Convert input to shape (B, C, T)."""
        if x.ndim == 3:
            if x.shape[1] == self.n_channels:
                pass  
            elif x.shape[2] == self.n_channels:
                x = x.transpose(1, 2)  
            else:
                raise ValueError(
                    "3D input must have shape (B, C, T) or (B, T, C). "
                    f"Got shape {tuple(x.shape)} with expected n_channels={self.n_channels}."
                )

        elif x.ndim == 4:
            if x.shape[1] != 1:
                raise ValueError(
                    "4D input must have shape (B, 1, C, T) or (B, 1, T, C). "
                    f"Got shape {tuple(x.shape)}."
                )

            if x.shape[2] == self.n_channels:
                x = x.squeeze(1)  
            elif x.shape[3] == self.n_channels:
                x = x.squeeze(1).transpose(1, 2)  
            else:
                raise ValueError(
                    "4D input must have channels in dim=2 or dim=3. "
                    f"Got shape {tuple(x.shape)} with expected n_channels={self.n_channels}."
                )
        else:
            raise ValueError(
                "Input must have shape (B, C, T), (B, T, C), (B, 1, C, T), or (B, 1, T, C). "
                f"Got {x.ndim} dimensions."
            )

        if x.shape[1] != self.n_channels:
            raise ValueError(
                f"Expected {self.n_channels} channels after normalization, but got {x.shape[1]}."
            )

        return x

    def _extract_last_valid_timestep(
        self,
        output: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get the last valid LSTM output for each sequence.

        Args:
            output: (B, T', H)
            lengths: optional tensor of shape (B,) with valid sequence lengths
                     after CNN+pooling. If None, uses the final timestep.

        Returns:
            Tensor of shape (B, H)
        """
        if lengths is None:
            return output[:, -1, :]

        if lengths.ndim != 1 or lengths.shape[0] != output.shape[0]:
            raise ValueError(
                f"lengths must have shape ({output.shape[0]},), got {tuple(lengths.shape)}"
            )

        idx = (lengths - 1).clamp(min=0)
        batch_idx = torch.arange(output.shape[0], device=output.device)
        return output[batch_idx, idx, :]

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: EEG input
            lengths: optional original valid lengths before CNN pooling.
                     Only needed if you use padded variable-length batches.

        Returns:
            logits: (B, n_classes)
        """
        x = self._normalize_input_shape(x)  

        if lengths is not None:
            if not torch.is_tensor(lengths):
                raise TypeError("lengths must be a torch.Tensor when provided")
            lengths = lengths.to(device=x.device)

        x = self.cnn(x)                   
        x = x.transpose(1, 2)             

        if lengths is not None:
            lengths = torch.div(lengths, 2, rounding_mode="floor")
            lengths = torch.div(lengths, 2, rounding_mode="floor")
            lengths = lengths.clamp(min=1, max=x.shape[1])

            packed = nn.utils.rnn.pack_padded_sequence(
                x,
                lengths.cpu(),
                batch_first=True,
                enforce_sorted=False,
            )
            packed_out, _ = self.lstm(packed)
            x, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out,
                batch_first=True,
            )
            x = self._extract_last_valid_timestep(x, lengths)
        else:
            x, _ = self.lstm(x)              
            x = x[:, -1, :]                 

        logits = self.classifier(x)        
        return logits