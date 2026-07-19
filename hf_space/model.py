import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


class CrossStageResNet(nn.Module):
    def __init__(self):
        super().__init__()
        # The complete trained state is loaded immediately after construction.
        # Avoid an unnecessary ImageNet download in the hosted runtime.
        resnet = tv_models.resnet18(weights=None)
        self.stem_to_l3 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
        )
        self.layer4 = resnet.layer4
        self.downsample_l3 = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, inputs):
        stage_three = self.stem_to_l3(inputs)
        stage_four = self.layer4(stage_three)
        return torch.cat([self.downsample_l3(stage_three), stage_four], dim=1)


class PartAttentionAndGating(nn.Module):
    def __init__(self, in_channels=768, num_parts=4):
        super().__init__()
        self.num_parts = num_parts
        self.channel_dropout = nn.Dropout2d(p=0.15)
        self.attention_conv = nn.Conv2d(in_channels, num_parts, kernel_size=1)
        self.gating_fc = nn.ModuleList(
            [nn.Linear(in_channels, 1) for _ in range(num_parts)]
        )
        self.temperature = 4.0

    def forward(self, feature_map, foreground_mask):
        batch_size, _, height, width = feature_map.shape
        logits = self.attention_conv(self.channel_dropout(feature_map))
        logits = logits.view(batch_size, self.num_parts, -1) / self.temperature
        valid = foreground_mask.view(batch_size, -1) > 0.05
        empty = valid.sum(dim=1) == 0
        if empty.any():
            valid = valid.clone()
            valid[empty] = True
        logits = logits.masked_fill(~valid.unsqueeze(1), -1e4)
        attention = F.softmax(logits, dim=-1).view(
            batch_size, self.num_parts, height, width
        )

        parts = []
        gates = []
        for index in range(self.num_parts):
            mask = attention[:, index].unsqueeze(1)
            feature = (feature_map * mask).sum(dim=(2, 3))
            gate = torch.sigmoid(self.gating_fc[index](feature))
            parts.append(feature * gate)
            gates.append(gate)
        return torch.cat(parts, dim=1), attention, gates


class TemperamentOmicsNet(nn.Module):
    def __init__(self, feature_dim=768, num_parts=4):
        super().__init__()
        self.backbone = CrossStageResNet()
        self.part_fusion = PartAttentionAndGating(feature_dim, num_parts)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.projector = nn.Sequential(
            nn.Linear(feature_dim + feature_dim * num_parts, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(512, 128),
        )

    def forward(self, inputs, foreground_mask, return_masks=False):
        feature_map = self.backbone(inputs)
        global_feature = self.global_pool(feature_map).flatten(1)
        pooled_mask = F.adaptive_avg_pool2d(
            foreground_mask.float(), feature_map.shape[-2:]
        )
        hard_foreground = (pooled_mask > 0.10).float()
        part_features, masks, gates = self.part_fusion(
            feature_map, hard_foreground
        )
        embedding = F.normalize(
            self.projector(torch.cat([global_feature, part_features], dim=1)),
            p=2,
            dim=1,
        )
        if return_masks:
            return embedding, masks, gates
        return embedding
