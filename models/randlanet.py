#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/7/6 11:36
# @File    : randlanet.py
# @Description :

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.neighbors import KDTree

from utils.lovasz_losses import lovasz_softmax_flat


class AttentivePooling(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.fc = nn.Conv2d(d_in, d_in, (1, 1), bias=False)

        self.mlp = nn.Sequential(
            nn.Conv2d(d_in, d_out, 1),
            nn.BatchNorm2d(d_out),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

    def forward(self, feature_set):
        att_activation = self.fc(feature_set)
        att_scores = F.softmax(att_activation, dim=3)
        f_agg = feature_set * att_scores
        f_agg = torch.sum(f_agg, dim=3, keepdim=True)
        f_agg = self.mlp(f_agg)
        return f_agg


class Building_block(nn.Module):
    def __init__(self, d_out):  # d_in = d_out//2
        super().__init__()

        self.mlp1 = nn.Sequential(
            nn.Conv2d(10, d_out//2, 1),
            nn.BatchNorm2d(d_out//2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.att_pooling_1 = AttentivePooling(d_out, d_out//2)
        self.mlp2 = nn.Sequential(
            nn.Conv2d(d_out//2, d_out//2, 1),
            nn.BatchNorm2d(d_out//2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.att_pooling_2 = AttentivePooling(d_out, d_out)

    def forward(self, xyz, feature, neigh_idx):  # feature: Batch*channel*npoints*1
        # xyz (B, N, 3)   neigh_idx(B, N, k)
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)  # batch*npoint*nsamples*10
        f_xyz = f_xyz.permute((0, 3, 1, 2)).contiguous()   # batch*10*npoint*nsamples
        f_xyz = self.mlp1(f_xyz)
        # batch*npoint*nsamples*channel
        f_neighbours = self.gather_neighbour(feature.squeeze(-1).permute((0, 2, 1)), neigh_idx)
        f_neighbours = f_neighbours.permute((0, 3, 1, 2)).contiguous()  # batch*channel*npoint*nsamples
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_1(f_concat)  # Batch*channel*npoints*1

        f_xyz = self.mlp2(f_xyz)
        # batch*npoint*nsamples*channel
        f_neighbours = self.gather_neighbour(f_pc_agg.squeeze(-1).permute((0, 2, 1)), neigh_idx)
        f_neighbours = f_neighbours.permute((0, 3, 1, 2)).contiguous()  # batch*channel*npoint*nsamples
        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)
        f_pc_agg = self.att_pooling_2(f_concat)
        return f_pc_agg

    def relative_pos_encoding(self, xyz, neigh_idx):
        neighbor_xyz = self.gather_neighbour(xyz, neigh_idx)  # batch*npoint*nsamples*3

        xyz_tile = xyz.unsqueeze(2).expand(-1, -1, neigh_idx.shape[-1], -1)  # batch*npoint*nsamples*3
        relative_xyz = xyz_tile - neighbor_xyz  # batch*npoint*nsamples*3
        # batch*npoint*nsamples*1
        relative_dis = torch.norm(relative_xyz, dim=-1, keepdim=True)
        # batch*npoint*nsamples*10
        relative_feature = torch.cat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], dim=-1)
        return relative_feature

    @staticmethod
    def gather_neighbour(pc, neighbor_idx):  # pc: batch*npoint*channel
        # gather the coordinates or features of neighboring points
        batch_size = pc.shape[0]
        num_points = pc.shape[1]
        d = pc.shape[2]
        index_input = neighbor_idx.view(batch_size, -1)
        features = torch.gather(pc, 1, index_input.unsqueeze(-1).expand(-1, -1, pc.shape[2]))
        features = features.view(batch_size, num_points, neighbor_idx.shape[-1], d)  # batch*npoint*nsamples*channel
        return features


class DilatedResidualBlock(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.mlp1 = nn.Sequential(
            nn.Conv2d(d_in, d_out // 2, 1),
            nn.BatchNorm2d(d_out // 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.lfa = Building_block(d_out)
        self.mlp2 = nn.Sequential(
            nn.Conv2d(d_out, d_out*2, 1),
            nn.BatchNorm2d(d_out*2),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(d_in, d_out*2, 1),
            nn.BatchNorm2d(d_out*2),
        )

    def forward(self, feature, xyz, neigh_idx):
        f_pc = self.mlp1(feature)  # Batch*channel*npoints*1
        f_pc = self.lfa(xyz, f_pc, neigh_idx)  # Batch*d_out*npoints*1
        f_pc = self.mlp2(f_pc)
        shortcut = self.shortcut(feature)
        return F.leaky_relu(f_pc+shortcut, negative_slope=0.2)


class get_model(nn.Module):

    def __init__(self, d_in, num_classes, num_neighbors=16):
        super().__init__()
        self.num_layers = 5
        self.sub_sampling_ratio = [4, 4, 4, 4, 2]
        self.d_out = [16, 64, 128, 256, 512]
        self.k_n = num_neighbors
        self.fc0 = nn.Sequential(
            nn.Conv1d(d_in, 8, 1),
            nn.BatchNorm1d(8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.dilated_res_blocks = nn.ModuleList()
        d_in = 8
        for i in range(self.num_layers):
            d_out = self.d_out[i]
            # DilatedResidualBlock模块返回的feature的C 是2*d_out
            self.dilated_res_blocks.append(DilatedResidualBlock(d_in, d_out))
            d_in = 2 * d_out

        d_out = d_in
        self.decoder_0 = nn.Sequential(
            nn.Conv2d(d_in, d_out, 1),
            nn.BatchNorm2d(d_out),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.decoder_blocks = nn.ModuleList()
        for j in range(self.num_layers):
            if j <= 3:
                d_in = d_out + 2 * self.d_out[-j-2]
                d_out = 2 * self.d_out[-j-2]
            else:
                d_in = 4 * self.d_out[-j-1]
                d_out = 2 * self.d_out[-j-1]
            self.decoder_blocks.append(nn.Sequential(
                nn.Conv2d(d_in, d_out, 1),
                nn.BatchNorm2d(d_out),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ))

        self.fc1 = nn.Sequential(
            nn.Conv2d(d_out, 64, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(64, 32, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.dropout = nn.Dropout(0.5)
        self.fc3 = nn.Conv2d(32, num_classes, kernel_size=(1, 1))

    def forward(self, points):
        end_points = self.compute_indices(points)
        features = end_points['features']
        features = self.fc0(features)

        features = features.unsqueeze(dim=3)  # Batch*channel*npoints*1

        # ###########################Encoder############################
        f_encoder_list = []
        for i in range(self.num_layers):
            f_encoder_i = self.dilated_res_blocks[i](features, end_points['xyz'][i], end_points['neigh_idx'][i])

            f_sampled_i = self.random_sample(f_encoder_i, end_points['sub_idx'][i])
            features = f_sampled_i
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
        # ###########################Encoder############################

        features = self.decoder_0(f_encoder_list[-1])
        # print(len(f_encoder_list))
        # ###########################Decoder############################
        f_decoder_list = []
        for j in range(self.num_layers):
            f_interp_i = self.nearest_interpolation(features, end_points['interp_idx'][-j - 1])
            # print(f"{j} -- {f_encoder_list[-j-2].shape} -- {f_interp_i.shape}")
            f_decoder_i = self.decoder_blocks[j](torch.cat([f_encoder_list[-j - 2], f_interp_i], dim=1))

            features = f_decoder_i
            f_decoder_list.append(f_decoder_i)
        # ###########################Decoder############################

        features = self.fc1(features)
        features = self.fc2(features)
        features = self.dropout(features)
        features = self.fc3(features)
        f_out = features.squeeze(3)
        output = F.log_softmax(f_out, dim=1)
        output = output.permute(0, 2, 1).contiguous()
        # end_points['logits'] = f_out
        return output


    def compute_indices(self, points):
        r"""
        修改自定义数据集中的tf_map方法，将得到的：下采样点、knn. 池化索引. 上采样索引
        Args:
            points: point of Shape(B, C, N)

        Returns:
             end_points['xyz'][i]
             end_points['neigh_idx'][i]
             end_points['sub_idx'][i]
             end_points['interp_idx'][-j - 1]
        """
        device = points.device
        batch_size, channels, num_points = points.shape
        xyz = points[:, :3, :].permute(0, 2, 1).contiguous()
        input_points = []
        input_neighbors = []
        input_pools = []
        input_up_samples = []

        for i in range(self.num_layers):
            neighbor_idx = torch.from_numpy(self.knn_search(xyz.cpu().numpy(), xyz.cpu().numpy(), self.k_n)).long().to(device)
            # neighbor_idx = self.knn_search(xyz, xyz, self.k_n)
            sub_points = xyz[:, :xyz.shape[1] // self.sub_sampling_ratio[i], :]
            pool_i = neighbor_idx[:, :xyz.shape[1] // self.sub_sampling_ratio[i], :]
            # up_i = self.knn_search(sub_points, xyz, 1)
            up_i = torch.from_numpy(self.knn_search(sub_points.cpu().numpy(), xyz.cpu().numpy(), 1)).long().to(device)

            # input_points.append(torch.from_numpy(xyz).float())
            # input_neighbors.append(torch.from_numpy(neighbor_idx).long())
            # input_pools.append(torch.from_numpy(pool_i).long())
            # input_up_samples.append(torch.from_numpy(up_i).long())
            input_points.append(xyz)
            input_neighbors.append(neighbor_idx)
            input_pools.append(pool_i)
            input_up_samples.append(up_i)
            xyz = sub_points

        input_list = input_points + input_neighbors + input_pools + input_up_samples
        input_list += [points]

        num_layers = self.num_layers
        inputs = {}
        inputs['xyz'] = input_list[:num_layers]
        inputs['neigh_idx'] = input_list[num_layers: 2 * num_layers]
        inputs['sub_idx'] = input_list[2 * num_layers:3 * num_layers]
        inputs['interp_idx'] = input_list[3 * num_layers:4 * num_layers]
        inputs['features'] = input_list[4 * num_layers]

        return inputs

    @staticmethod
    def knn_search(support_pts, query_pts, k):
        batch_size, num_support_points, _ = support_pts.shape
        _, num_query_points, _ = query_pts.shape
        neighbor_idx = np.zeros((batch_size, num_query_points, k), dtype=np.int64)

        for i in range(batch_size):
            kdtree = KDTree(support_pts[i])
            _, idx = kdtree.query(query_pts[i], k)
            neighbor_idx[i] = idx

        return neighbor_idx

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, N, d] input features matrix
        :param pool_idx: [B, N', max_num] N' < N, N' is the selected position after pooling
        :return: pool_features = [B, N', d] pooled features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        num_neigh = pool_idx.shape[-1]
        d = feature.shape[1]
        batch_size = pool_idx.shape[0]
        pool_idx = pool_idx.view(batch_size, -1)  # batch*(npoints,nsamples)
        pool_features = torch.gather(feature, 2, pool_idx.unsqueeze(1).expand(-1, feature.shape[1], -1))
        pool_features = pool_features.view(batch_size, d, -1, num_neigh)
        pool_features = pool_features.max(dim=3, keepdim=True)[0]  # batch*channel*npoints*1
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, d, N] input features matrix
        :param interp_idx: [B, up_num_points, 1] nearest neighbour index
        :return: [B, d, up_num_points] interpolated features matrix
        """
        feature = feature.squeeze(dim=3)  # batch*channel*npoints
        batch_size = interp_idx.shape[0]
        up_num_points = interp_idx.shape[1]
        interp_idx = interp_idx.view(batch_size, up_num_points)
        interpolated_features = torch.gather(feature, 2, interp_idx.unsqueeze(1).expand(-1, feature.shape[1], -1))
        interpolated_features = interpolated_features.unsqueeze(3)  # batch*channel*npoints*1
        return interpolated_features


class get_loss(nn.Module):
    def __init__(self, weight, weight_ce=1.0, weight_lovasz=1.0):
        super(get_loss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_lovasz = weight_lovasz
        self.ce_loss = nn.CrossEntropyLoss(weight=weight)

    def forward(self, logits, labels):
        # logits: [B*N, C]
        # labels: [B*N]

        # Calculate CrossEntropyLoss
        ce_loss = self.ce_loss(logits, labels)

        # Calculate Lovasz-Softmax Loss
        probas = torch.nn.functional.softmax(logits, dim=1)
        lovasz_loss = lovasz_softmax_flat(probas, labels)

        # Weighted combination of losses
        total_loss = self.weight_ce * ce_loss + self.weight_lovasz * lovasz_loss
        return total_loss

# class get_loss(nn.Module):
#     def __init__(self):
#         super(get_loss, self).__init__()
#
#     def forward(self, pred, target, weight):
#         # print("Target:", target)
#         # print("Weights:", weight)
#         total_loss = F.nll_loss(pred, target, weight=weight)
#         # print("Loss:", total_loss.item())  # 添加调试信息
#         return total_loss


# Example usage
# num_classes = 13  # For S3DIS dataset
# model = get_model(9, num_classes, num_neighbors=16, decimation=4)
#
# # Assuming `data` is your input point cloud data with shape (batch_size, num_points, num_features)
# # Example:
# data = torch.randn(32, 9, 4096)
# output = model(data)
# print(output.shape)