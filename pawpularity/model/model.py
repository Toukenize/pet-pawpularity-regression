import timm
import torch
import torch.nn as nn


class PawImgModel(nn.Module):

    def __init__(self, model_name, dropout, meta_col_dim=0, **params):

        super().__init__()
        self.model = timm.create_model(model_name, **params)
        self.dropout1 = nn.Dropout(dropout)

        # Append a new linear layer on top of swin model (including imagenet head)
        self.meta_col_dim = meta_col_dim
        lin_w_meta = 1_000 + meta_col_dim
        self.out = nn.Linear(lin_w_meta, 1)

    def forward(self, image, meta):
        img_out = self.model(image)
        img_out = self.dropout1(img_out)

        if self.meta_col_dim > 0:
            out = torch.cat([img_out, meta], dim=-1)
        else:
            out = img_out

        out = self.out(out)

        return out
