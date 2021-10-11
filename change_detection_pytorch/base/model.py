import torch
import torch.nn.functional as F
from . import initialization as init


class SegmentationModel(torch.nn.Module):

    def initialize(self):
        init.initialize_decoder(self.decoder)
        init.initialize_head(self.segmentation_head)
        if self.classification_head is not None:
            init.initialize_head(self.classification_head)

    def base_forward(self, x1, x2):
        """Sequentially pass `x1` `x2` trough model`s encoder, decoder and heads"""
        if self.siam_encoder:
            features = self.encoder(x1), self.encoder(x2)
        else:
            features = self.encoder(x1), self.encoder_non_siam(x2)

        decoder_output = self.decoder(*features)

        try:
            masks = self.segmentation_head(decoder_output)
        except:
            try:
                if self.cond:
                    masks = []
                    [masks.extend(self.segmentation_head[i](decoder_output[i])) for i in range(len(decoder_output))]
            except:
                masks = [self.segmentation_head[i](decoder_output[i]) for i in range(len(decoder_output))]

        if self.classification_head is not None:
            labels = self.classification_head(features[-1])
            return masks, labels

        return masks

    def forward(self, x1, x2, TTA=False):
        """Sequentially pass `x1` `x2` trough model`s encoder, decoder and heads"""
        if not TTA:
            return self.base_forward(x1, x2)
        else:
            out = self.base_forward(x1, x2)
            out = F.softmax(out, dim=1)
            origin_x1 = x1.clone()
            origin_x2 = x2.clone()

            x1 = origin_x1.flip(2)
            x2 = origin_x2.flip(2)
            cur_out = self.base_forward(x1, x2)
            out += F.softmax(cur_out, dim=1).flip(2)

            x1 = origin_x1.flip(3)
            x2 = origin_x2.flip(3)
            cur_out = self.base_forward(x1, x2)
            out += F.softmax(cur_out, dim=1).flip(3)

            x1 = origin_x1.transpose(2, 3).flip(3)
            x2 = origin_x2.transpose(2, 3).flip(3)
            cur_out = self.base_forward(x1, x2)
            out += F.softmax(cur_out, dim=1).flip(3).transpose(2, 3)

            x1 = origin_x1.flip(3).transpose(2, 3)
            x2 = origin_x2.flip(3).transpose(2, 3)
            cur_out = self.base_forward(x1, x2)
            out += F.softmax(cur_out, dim=1).transpose(2, 3).flip(3)

            x1 = origin_x1.flip(2).flip(3)
            x2 = origin_x2.flip(2).flip(3)
            cur_out = self.base_forward(x1, x2)
            out += F.softmax(cur_out, dim=1).flip(3).flip(2)

            out /= 6.0

            return out

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)` with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)

        return x
