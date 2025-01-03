from .swintransformerv2 import SwinTransformerV2
import torch.nn as nn
import torch
from segmentation_models_pytorch.base import modules as md
import torch.nn.functional as F
from .cbam import CbamModule
from torchvision import models
from itertools import chain



def swin_v2(size, img_size=256, in_22k=False, config=None, pretrained=False, device='cuda', **kwargs):
    if size == "swinv2_tiny_window16_256":
        model = SwinTransformerV2(img_size=img_size, window_size=16, embed_dim=96, depths=[2, 2, 6, 2],
                                  num_heads=[3, 6, 12, 24], **kwargs)
        if pretrained:
            checkpoint = torch.load(config.model_urls[size])["model"]
            if img_size != 256:
                del checkpoint["layers.0.blocks.0.attn.relative_coords_table"]
                del checkpoint["layers.0.blocks.0.attn.relative_position_index"]
                del checkpoint["layers.0.blocks.1.attn_mask"]
                del checkpoint["layers.0.blocks.1.attn.relative_coords_table"]
                del checkpoint["layers.0.blocks.1.attn.relative_position_index"]
                del checkpoint["layers.1.blocks.0.attn.relative_coords_table"]
                del checkpoint["layers.1.blocks.0.attn.relative_position_index"]
                del checkpoint["layers.1.blocks.1.attn_mask"]
                del checkpoint["layers.1.blocks.1.attn.relative_coords_table"]
                del checkpoint["layers.1.blocks.1.attn.relative_position_index"]
                del checkpoint["layers.2.blocks.0.attn.relative_coords_table"]
                del checkpoint["layers.2.blocks.0.attn.relative_position_index"]
                del checkpoint["layers.2.blocks.1.attn_mask"]
                del checkpoint["layers.2.blocks.1.attn.relative_coords_table"]
                del checkpoint["layers.2.blocks.1.attn.relative_position_index"]
                del checkpoint["layers.2.blocks.2.attn.relative_coords_table"]
                del checkpoint["layers.2.blocks.2.attn.relative_position_index"]
                del checkpoint["layers.2.blocks.3.attn_mask"]
                del checkpoint["layers.2.blocks.3.attn.relative_coords_table"]
                del checkpoint["layers.2.blocks.3.attn.relative_position_index"]
                del checkpoint["layers.2.blocks.4.attn.relative_coords_table"]
                del checkpoint["layers.2.blocks.4.attn.relative_position_index"]
                del checkpoint["layers.2.blocks.5.attn_mask"]
                del checkpoint["layers.2.blocks.5.attn.relative_coords_table"]
                del checkpoint["layers.2.blocks.5.attn.relative_position_index"]
                del checkpoint["layers.3.blocks.0.attn.relative_coords_table"]
                del checkpoint["layers.3.blocks.0.attn.relative_position_index"]
                del checkpoint["layers.3.blocks.1.attn.relative_coords_table"]
                del checkpoint["layers.3.blocks.1.attn.relative_position_index"]
            model.load_state_dict(checkpoint, strict=False)
        else:
            pass


    elif size == "swinv2_small_window8_256":
        model = SwinTransformerV2(img_size=img_size, window_size=8, embed_dim=96, depths=[2, 2, 18, 2],
                                  num_heads=[3, 6, 12, 24], **kwargs)

        if pretrained:
            checkpoint = torch.load(config.model_urls[size])["model"]
            if img_size != 256:
                del checkpoint["layers.0.blocks.1.attn_mask"]
                del checkpoint["layers.1.blocks.1.attn_mask"]
                del checkpoint["layers.2.blocks.1.attn_mask"]
                del checkpoint["layers.2.blocks.3.attn_mask"]
                del checkpoint["layers.2.blocks.5.attn_mask"]
                del checkpoint["layers.2.blocks.7.attn_mask"]
                del checkpoint["layers.2.blocks.9.attn_mask"]
                del checkpoint["layers.2.blocks.11.attn_mask"]
                del checkpoint["layers.2.blocks.13.attn_mask"]
                del checkpoint["layers.2.blocks.15.attn_mask"]
                del checkpoint["layers.2.blocks.17.attn_mask"]

            model.load_state_dict(checkpoint, strict=False)

        else:
            pass

    elif size == "swinv2_small_window16_256":
        model = SwinTransformerV2(img_size=img_size, window_size=16, embed_dim=96, depths=[2, 2, 18, 2],
                                  num_heads=[3, 6, 12, 24], **kwargs)
        if pretrained:
            checkpoint = torch.load(config.model_urls[size])["model"]
            if img_size != 256:
                del checkpoint["layers.0.blocks.1.attn_mask"]
                del checkpoint["layers.1.blocks.1.attn_mask"]
                del checkpoint["layers.3.blocks.0.attn.relative_coords_table"]
                del checkpoint["layers.3.blocks.0.attn.relative_position_index"]
                del checkpoint["layers.3.blocks.1.attn.relative_coords_table"]
                del checkpoint["layers.3.blocks.1.attn.relative_position_index"]
            model.load_state_dict(checkpoint, strict=False)
        else:
            pass
    elif size == "swinv2_base_window16_256":
        model = SwinTransformerV2(img_size=img_size, window_size=16, embed_dim=128, depths=[2, 2, 18, 2],
                                  num_heads=[4, 8, 16, 32], **kwargs)
        if pretrained:
            checkpoint = torch.load(config.model_urls[size])["model"]
            if img_size != 256:
                del checkpoint["layers.0.blocks.1.attn_mask"]
                del checkpoint["layers.1.blocks.1.attn_mask"]
                del checkpoint["layers.3.blocks.0.attn.relative_coords_table"]
                del checkpoint["layers.3.blocks.0.attn.relative_position_index"]
                del checkpoint["layers.3.blocks.1.attn.relative_coords_table"]
                del checkpoint["layers.3.blocks.1.attn.relative_position_index"]
            model.load_state_dict(checkpoint, strict=False)
        else:
            pass

    model = model.to(device)
    return model


class PSPModule(nn.Module):
    # In the original inmplementation they use precise RoI pooling
    # Instead of using adaptative average pooling
    def __init__(self, in_channels, bin_sizes=[1, 2, 4, 6]):
        super(PSPModule, self).__init__()
        out_channels = in_channels // len(bin_sizes)
        self.stages = nn.ModuleList([self._make_stages(in_channels, out_channels, b_s)
                                     for b_s in bin_sizes])
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels + (out_channels * len(bin_sizes)), in_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1)
        )

    def _make_stages(self, in_channels, out_channels, bin_sz):
        prior = nn.AdaptiveAvgPool2d(output_size=bin_sz)
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU(inplace=True)
        return nn.Sequential(prior, conv, bn, relu)

    def forward(self, features):
        h, w = features.size()[2], features.size()[3]
        pyramids = [features]
        pyramids.extend([F.interpolate(stage(features), size=(h, w), mode='bilinear',
                                       align_corners=True) for stage in self.stages])
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return output


class ResNet(nn.Module):
    def __init__(self, in_channels=3, output_stride=16, backbone='resnet101', pretrained=True):
        super(ResNet, self).__init__()
        model = getattr(models, backbone)(pretrained)
        if not pretrained or in_channels != 3:
            self.initial = nn.Sequential(
                nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )
            # initialize_weights(self.initial)
        else:
            self.initial = nn.Sequential(*list(model.children())[:4])

        self.layer1 = model.layer1
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4

        if output_stride == 16:
            s3, s4, d3, d4 = (2, 1, 1, 2)
        elif output_stride == 8:
            s3, s4, d3, d4 = (1, 1, 2, 4)

        if output_stride == 8:
            for n, m in self.layer3.named_modules():
                if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3, s3)
                elif 'conv2' in n:
                    m.dilation, m.padding, m.stride = (d3, d3), (d3, d3), (s3, s3)
                elif 'downsample.0' in n:
                    m.stride = (s3, s3)

        for n, m in self.layer4.named_modules():
            if 'conv1' in n and (backbone == 'resnet34' or backbone == 'resnet18'):
                m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
            elif 'conv2' in n:
                m.dilation, m.padding, m.stride = (d4, d4), (d4, d4), (s4, s4)
            elif 'downsample.0' in n:
                m.stride = (s4, s4)

    def forward(self, x):
        x = self.initial(x)
        x1 = self.layer1(x)
        print("x1", x1.shape)
        x2 = self.layer2(x1)
        print("x2", x2.shape)
        x3 = self.layer3(x2)
        print("x3", x3.shape)
        x4 = self.layer4(x3)
        print("x4", x4.shape)

        return [x1, x2, x3, x4]


def up_and_add(x, y):
    return F.interpolate(x, size=(y.size(2), y.size(3)), mode='bilinear', align_corners=True) + y


class FPN_fuse(nn.Module):
    def __init__(self, feature_channels=[256, 512, 1024, 2048], fpn_out=256):
        super(FPN_fuse, self).__init__()
        assert feature_channels[0] == fpn_out
        self.conv1x1 = nn.ModuleList([nn.Conv2d(ft_size, fpn_out, kernel_size=1)
                                      for ft_size in feature_channels[1:]])
        self.smooth_conv = nn.ModuleList([nn.Conv2d(fpn_out, fpn_out, kernel_size=3, padding=1)]
                                         * (len(feature_channels) - 1))
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(len(feature_channels) * fpn_out, fpn_out, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, features):
        features[1:] = [conv1x1(feature) for feature, conv1x1 in zip(features[1:], self.conv1x1)]  ##
        P = [up_and_add(features[i], features[i - 1]) for i in reversed(range(1, len(features)))]
        P = [smooth_conv(x) for smooth_conv, x in zip(self.smooth_conv, P)]
        P = list(reversed(P))
        P.append(features[-1])  # P = [P1, P2, P3, P4]
        H, W = P[0].size(2), P[0].size(3)
        P[1:] = [F.interpolate(feature, size=(H, W), mode='bilinear', align_corners=True) for feature in P[1:]]

        x = self.conv_fusion(torch.cat((P), dim=1))
        return x


class UperNet_swin(nn.Module):
    # Implementing only the object path
    def __init__(self, size="swinv2_small_window16_256", config=None, img_size=256, num_classes=1, in_channels=3,
                 pretrained=True):
        super(UperNet_swin, self).__init__()

        self.backbone = swin_v2(size=size, img_size=img_size, config=config)
        if size.split("_")[1] in ["small", "tiny"]:
            feature_channels = [192, 384, 768, 768]
        elif size.split("_")[1] in ["base"]:
            feature_channels = [256, 512, 1024, 1024]
        self.PPN = PSPModule(feature_channels[-1])
        self.FPN = FPN_fuse(feature_channels, fpn_out=feature_channels[0])
        self.head = nn.Conv2d(feature_channels[0], num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])

        features = self.backbone.extra_features(x)
        features[-1] = self.PPN(features[-1])
        x = self.head(self.FPN(features))

        x = F.interpolate(x, size=input_size, mode='bilinear')
        return x

    def get_backbone_params(self):
        return self.backbone.parameters()

    def get_decoder_params(self):
        return chain(self.PPN.parameters(), self.FPN.parameters(), self.head.parameters())

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()


class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        if attention_type == "cbam":
            self.attention1 = CbamModule(channels=in_channels + skip_channels)
        else:
            self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        if attention_type == "cbam":
            self.attention2 = CbamModule(channels=out_channels)
        else:
            self.attention2 = md.Attention(attention_type, in_channels=out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.skip_channels = skip_channels

    def forward(self, x, skip=None):
        if skip is None:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        else:
            if x.shape[-1] != skip.shape[-1]:
                x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            # print(x.shape,skip.shape)
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]

        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(head_channels, head_channels, use_batchnorm=use_batchnorm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        features = features[1:]  # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
            # y_i = self.upsample1(y_i)
        # hypercol = torch.cat([y0,y1,y2,y3,y4], dim=1)

        return x


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class unet_swin(nn.Module):
    def __init__(
            self, config, size="small", img_size=256  # "base" "large"
    ):
        super().__init__()

        self.encoder = swin_v2(size=size, img_size=img_size, config=config)

        if size.split("_")[1] in ["small", "tiny"]:
            feature_channels = (3, 192, 384, 768, 768)
        elif size.split("_")[1] in ["base"]:
            feature_channels = (3, 256, 512, 1024, 1024)
        self.decoder = UnetDecoder(encoder_channels=feature_channels, n_blocks=4, decoder_channels=(512, 256, 128, 64),
                                   attention_type=None)

        self.segmentation_head = SegmentationHead(in_channels=64, out_channels=1, kernel_size=3, upsampling=4
                                                  )

    def forward(self, input):
        encoder_featrue = self.encoder.get_unet_feature(input)
        decoder_output = self.decoder(*encoder_featrue)
        masks = self.segmentation_head(decoder_output)

        return masks