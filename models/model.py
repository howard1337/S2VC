"""FragmentVC model architecture."""

from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .convolutional_transformer import Smoother, Extractor

class S2VC(nn.Module):
    """
    FragmentVC uses Wav2Vec feature of the source speaker to query and attend
    on mel spectrogram of the target speaker.
    """

    def __init__(self, input_dim, ref_dim, d_model=512):
        super().__init__()
        self.unet = UnetBlock(d_model, input_dim, ref_dim)

        self.smoothers = nn.TransformerEncoder(Smoother(d_model, 2, 1024), num_layers=3)

        self.mel_linear = nn.Linear(d_model, 80)

        self.post_net = nn.Sequential(
            nn.Conv1d(80, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 512, kernel_size=5, padding=2),
            nn.BatchNorm1d(512),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Conv1d(512, 80, kernel_size=5, padding=2),
            nn.BatchNorm1d(80),
            nn.Dropout(0.5),
        )

    def forward(
        self,
        srcs: Tensor,
        refs: Tensor,
        src_masks: Optional[Tensor] = None,
        ref_masks: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Optional[Tensor]]]:
        """Forward function.

        Args:
            srcs: (batch, src_len, 768)
            src_masks: (batch, src_len)
            refs: (batch, 80, ref_len)
            ref_masks: (batch, ref_len)
        """
        # out: (src_len, batch, d_model)
        out, attns = self.unet(srcs, refs, src_masks=src_masks, ref_masks=ref_masks)

        # out: (src_len, batch, d_model)
        out = self.smoothers(out, src_key_padding_mask=src_masks)

        # out: (src_len, batch, 80)
        out = self.mel_linear(out)

        # out: (batch, 80, src_len)
        out = out.transpose(1, 0).transpose(2, 1)
        refined = self.post_net(out)
        out = out + refined

        # out: (batch, 80, src_len)
        return out, attns



class SelfAttentionPooling(nn.Module):
  """
  Implementation of SelfAttentionPooling from https://gist.github.com/pohanchi/c77f6dbfbcbc21c5215acde4f62e4362
  Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
  https://arxiv.org/pdf/2008.01077v1.pdf
  """
  def __init__(self, input_dim: int):
    super(SelfAttentionPooling, self).__init__()
    self.W = nn.Linear(input_dim, 1)
    self.softmax = nn.functional.softmax

  def forward(self, batch_rep: Tensor, att_mask: Optional[Tensor] = None):
    """
      N: batch size, T: sequence length, H: Hidden dimension
      input:
        batch_rep : size (N, T, H)
      attention_weight:
        att_w : size (N, T, 1)
      return:
        utter_rep: size (N, H)
    """
    att_logits = self.W(batch_rep).squeeze(-1)
    if att_mask is not None:
      att_logits = att_logits.masked_fill(att_mask, 1e-20)
    att_w = self.softmax(att_logits, dim=-1).unsqueeze(-1)
    utter_rep = torch.sum(batch_rep * att_w, dim=1)

    return utter_rep

class SourceEncoder(nn.Module):
    def __init__(self, d_model: int, input_dim: int):
        super(SourceEncoder, self).__init__()
        # encoder_layer = nn.TransformerEncoderLayer(d_model, 2, 1024, 0.1)
        # self.encoder = nn.TransformerEncoder(encoder_layer, 6)
        
        self.lin1 = nn.Linear(input_dim, input_dim)
        self.lin2 = nn.Linear(input_dim,  d_model)
        self.lin3 = nn.Linear(d_model,  d_model)
        self.lin4 = nn.Linear(d_model,  d_model)

        self.bn1 = nn.BatchNorm1d(input_dim)
        self.bn2 = nn.BatchNorm1d(d_model)
        self.bn3 = nn.BatchNorm1d(d_model)
        self.bn4 = nn.BatchNorm1d(d_model)

        self.dropout1 = nn.Dropout(0.0)
        self.dropout2 = nn.Dropout(0.0)
        self.dropout3 = nn.Dropout(0.0)
        self.dropout4 = nn.Dropout(0.0)

        self.SAP = SelfAttentionPooling(d_model)
        self.proj = nn.Linear(d_model, d_model)
        torch.nn.init.xavier_uniform_(
            self.proj.weight,   gain=torch.nn.init.calculate_gain('linear')
        )

    def forward(self, srcs: Tensor, refs: Tensor, src_masks: Optional[Tensor] = None, ref_masks: Optional[Tensor] = None):
        tgt = F.relu(self.lin1(srcs)).transpose(1, 2)
        tgt = self.dropout1(self.bn1(tgt)).transpose(1, 2)

        tgt = F.relu(self.lin2(tgt)).transpose(1, 2)
        tgt = self.dropout2(self.bn2(tgt)).transpose(1, 2)

        tgt = F.relu(self.lin3(tgt)).transpose(1, 2)
        tgt = self.dropout3(self.bn3(tgt)).transpose(1, 2)

        tgt = F.relu(self.lin4(tgt)).transpose(1, 2)
        tgt = self.dropout4(self.bn4(tgt)).transpose(1, 2)

        spk_embed = F.relu(self.proj(self.SAP(refs.transpose(1, 2), ref_masks))).unsqueeze(1)
        tgt *= spk_embed

        # tgt = self.encoder(tgt, src_masks)
        return tgt


class UnetBlock(nn.Module):
    """Hierarchically attend on references."""

    def __init__(self, d_model: int, input_dim: int, ref_dim: int):
        super(UnetBlock, self).__init__()
        self.conv1 = nn.Conv1d(ref_dim, d_model, 3, padding=1, padding_mode="replicate")
        self.conv2 = nn.Conv1d(d_model, d_model, 3, padding=1, padding_mode="replicate")
        self.conv3 = nn.Conv1d(d_model, d_model, 3, padding=1, padding_mode="replicate")

        use_bottleneck = True
        bottleneck_dim = 4
        n_head = 2
        self.extractor1 = Extractor(
            d_model, n_head, 1024, bottleneck_dim, no_residual=True, bottleneck=use_bottleneck,
        )

        self.src_encoder = SourceEncoder(d_model, input_dim)
    def forward(
        self,
        srcs: Tensor,
        refs: Tensor,
        src_masks: Optional[Tensor] = None,
        ref_masks: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Optional[Tensor]]]:
        """Forward function.

        Args:
            srcs: (batch, 80, src_len)
            src_masks: (batch, src_len)
            refs: (batch, 80, ref_len)
            ref_masks: (batch, ref_len)
        """

        # tgt: (batch, mel_len, bottleneck_dim)
        
        # tgt: (tgt_len, batch, bottleneck_dim)

        # ref*: (batch, d_model, mel_len)
        ref1 = self.conv1(refs)
        ref2 = self.conv2(F.relu(ref1))
        ref3 = self.conv3(F.relu(ref2))

        tgt = self.src_encoder(srcs, ref3, src_masks, ref_masks)
        tgt = tgt.transpose(0, 1)

        # out*: (tgt_len, batch, d_model)
        out, attn1 = self.extractor1(
            tgt,
            ref3.transpose(1, 2).transpose(0, 1),
            tgt_key_padding_mask=src_masks,
            memory_key_padding_mask=ref_masks,
        )
        return out, [attn1]

