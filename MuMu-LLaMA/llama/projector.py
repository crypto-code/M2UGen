import torch
from torch import nn

class ProjectionLayer(nn.Module):
    """Layers used in mapping text embeddings to visual outputs."""

    def __init__(self, in_dim: int, out_dim: int, num_input_tokens: int = 1, num_output_tokens: int = 1):
        super().__init__()

        self.num_input_tokens = num_input_tokens
        self.num_output_tokens = num_output_tokens
        self.out_dim = out_dim

        hidden_dim = 512
        self.fc = nn.Linear(in_dim, hidden_dim)
        self.tfm = nn.Transformer(batch_first=True, norm_first=False,
                                    d_model=hidden_dim, num_encoder_layers=4, num_decoder_layers=4,
                                    dim_feedforward=hidden_dim * 4, dropout=0.0, nhead=4)
        self.model = nn.Linear(hidden_dim, out_dim)
        self.query_embs = nn.Parameter(torch.randn(1, num_output_tokens, hidden_dim))

    def forward(self, x: torch.Tensor, input_embs: torch.Tensor) -> torch.Tensor:
        outputs = None
        x = x + input_embs
        x = self.fc(x)
        x = self.tfm(x, self.query_embs.repeat(x.shape[0], 1, 1))
        outputs = self.model(x)

        assert outputs.shape[1] == 1 or (
                    outputs.shape[1] * outputs.shape[2] == self.num_output_tokens * self.out_dim), (
            outputs.shape, self.num_output_tokens)
        return outputs  # (N, T_I_V_A.txt, D)