"""
@ author: neo
@ date: 2023-05-28  20:49 
@ file_name: attention.PY
@ github: https://github.com/Underson888/
"""

import numpy as np
import torch
from torch import nn
from models.containers import Module


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h):
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.fc_q.weight)
        nn.init.xavier_uniform_(self.fc_k.weight)
        nn.init.xavier_uniform_(self.fc_v.weight)
        nn.init.xavier_uniform_(self.fc_o.weight)
        nn.init.constant_(self.fc_q.bias, 0)
        nn.init.constant_(self.fc_k.bias, 0)
        nn.init.constant_(self.fc_v.bias, 0)
        nn.init.constant_(self.fc_o.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]
        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


class SummaryForgetAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, h,

                 *args, **kwargs):
        # 首先要对父类的参数进行初始化
        super().__init__(*args, **kwargs)

        # 传入超参数控制最后矩阵的形状,多头分解前
        self.in_forward_dimension = d_model
        self.keys_dimension = d_k
        self.queries_dimension = d_k
        self.values_dimension = d_v
        self.head_numbers = h
        # 该注意力内部需要的矩阵
        # self.fc_queries = nn.Linear(self.in_forward_dimension, self.head_numbers * self.queries_dimension)
        # self.fc_keys = nn.Linear(self.in_forward_dimension, self.head_numbers * self.keys_dimension)
        # self.fc_values = nn.Linear(self.in_forward_dimension, self.head_numbers * self.values_dimension)
        # 最后的输出使用的全连接层
        # 在transformer的注意力网络中，经过attention层的特征维度不发生变化
        # 这里不知道为啥是 values_dimension
        self.fc_output = nn.Linear(self.head_numbers * self.values_dimension, self.in_forward_dimension)
        # 这里模仿spring设置了一个放置layer的容器
        self.layer_context = {}

        # 用于细节微调的模块定义

        # 加入遗忘网络和总结模块
        # 设置遗忘模块大小
        self.after_forget_dimensions = self.in_forward_dimension // 2
        # 设置总结模块大小
        self.after_summary_dimensions = self.in_forward_dimension // 2
        # 设置遗忘模块
        self.keys_forget_weight = nn.Linear(self.in_forward_dimension, self.after_forget_dimensions)
        self.values_forget_weight = nn.Linear(self.in_forward_dimension, self.after_forget_dimensions)

        self.keys_forget_bias = nn.Parameter(torch.FloatTensor(1, 1, self.after_forget_dimensions))
        self.values_forget_bias = nn.Parameter(torch.FloatTensor(1, 1, self.after_forget_dimensions))

        # 设置总结单元
        self.keys_summary_unit = nn.Parameter(torch.FloatTensor(1, 1, self.after_summary_dimensions))
        self.values_summary_unit = nn.Parameter(torch.FloatTensor(1, 1, self.after_summary_dimensions))

        self.add_layers_into_context()

        self.init_attention_weights()

    def add_layers_into_context(self):
        # self.layer_context['fc'] = [self.fc_values, self.fc_keys, self.fc_queries, self.fc_output]
        self.layer_context['fc'] = [self.fc_output]
        self.layer_context['summary_modules'] = [self.keys_summary_unit, self.values_summary_unit]
        self.layer_context['forget_modules_weight'] = [self.keys_forget_weight, self.values_forget_weight]
        self.layer_context['forget_modules_bias'] = [self.keys_forget_bias, self.values_forget_bias]

    def init_attention_weights(self):
        # 这里对attention定义的参数进行初始化
        # 首先对权值进行初始化
        for fc_layer in self.layer_context['fc']:
            # Xavier initialization 对于使用sigmoid、tanh等对称和以0为中心的激活函数的网络尤其有效。然而对于ReLU激活函数，更常用的是Kaiming Initialization(
            # 或称为He Initialization)
            nn.init.xavier_uniform_(fc_layer.weight)
            # 使用0初始化偏置
            nn.init.constant_(fc_layer.bias, 0)
        for summary_module in self.layer_context['summary_modules']:
            nn.init.normal_(summary_module, 0, 1 / self.after_forget_dimensions)

        for forget_weight in self.layer_context['forget_modules_weight']:
            nn.init.xavier_uniform_(forget_weight.weight)
            nn.init.constant_(forget_weight.bias, 0)

        for forget_bias in self.layer_context['forget_modules_bias']:
            nn.init.normal_(forget_bias, 0, 1 / self.after_forget_dimensions)

    def forward(self, queries, keys, values, mask_for_attention_coefficient=None,
                weights_for_attention_coefficient=None):
        # 首先获取到各个维度
        batch_size, queries_electron_number, queries_dimensions = queries.shape
        keys_electron_number, keys_dimensions = keys.shape[1:]
        values_electron_number, values_dimensions = values.shape[1:]

        # 对values值进行遗忘
        after_forget_keys = torch.sigmoid(self.keys_forget_weight(keys) + self.keys_forget_bias)
        after_forget_values = torch.sigmoid(self.values_forget_weight(values) + self.values_forget_bias)

        # 进行总结
        # self.after_summary_dimensions (batch_size, keys_electron_number, after_summary_dimensions)
        summary_keys = self.keys_summary_unit.expand(batch_size, keys_electron_number, self.after_summary_dimensions)
        summary_values = self.values_summary_unit.expand(batch_size, values_electron_number,
                                                         self.after_summary_dimensions)
        # 总结和遗忘后内容进行结合
        updated_keys = torch.cat([after_forget_keys, summary_keys], dim=-1)
        updated_values = torch.cat([after_forget_values, summary_values], dim=-1)

        # 下面是注意力机制的计算
        # 将queries, keys, values投影到各自不同的空间
        # queries = self.fc_queries(queries)
        # keys = self.fc_keys(updated_keys)
        # values = self.fc_values(updated_values)

        # 分解为多头注意力形式，使用view函数
        # (batch_size, electron_number, head_numbers, dimensions)
        # (0, 1, 2, 3)
        queries = queries.view(batch_size, queries_electron_number, self.head_numbers, self.queries_dimension)
        keys = updated_keys.view(batch_size, keys_electron_number, self.head_numbers, self.keys_dimension)
        values = updated_values.view(batch_size, values_electron_number, self.head_numbers, self.values_dimension)

        # 交换维度，因为高维的矩阵乘法只能发生在最后两个维度
        # queries:(batch_size, head_numbers, electron_number, dimensions)
        # keys: (batch_size, head_numbers, dimensions, electron_number) 因为要计算的注意力系数是 queries * keys,所以转置的情况不同
        # values: (batch_size, head_numbers, electron_number, dimensions)
        # 这里也说明queries和keys的特征维度是相同的
        queries = queries.permute(0, 2, 1, 3)
        keys = keys.permute(0, 2, 3, 1)
        values = values.permute(0, 2, 1, 3)

        # 计算注意力系数
        # 这里一定用的是torch.matmul()的矩阵点积
        # attention_coefficient (batch_size, head_numbers, queries_electron_number, keys_electron_number)
        attention_coefficient = torch.matmul(queries, keys)
        # 对注意力系数标准化
        attention_coefficient = attention_coefficient / np.sqrt(self.keys_dimension)

        # 对于是否添加注意力权重判断
        if weights_for_attention_coefficient is not None:
            attention_coefficient = attention_coefficient * weights_for_attention_coefficient
        # 对是否mask进行判断
        if mask_for_attention_coefficient is not None:
            # 这里的mask_for_attention_coefficient必须是bool类型
            attention_coefficient = attention_coefficient.masked_fill_(mask_for_attention_coefficient, -np.inf)
        # 对计算的系数进行归一化
        attention_coefficient = torch.softmax(attention_coefficient, -1)
        # 将查询的注意力系数与values作用以更新
        # 这里说明keys和values的electron_number必须相同
        # 上面说明queries和keys的electron_dimensions是相同的
        # queries的electron_dimensions可以和values的不同
        # updated_values (batch_size, head_numbers, queries_electron_number, keys_electron_dimensions)
        updated_queries = torch.matmul(attention_coefficient, values)
        # 将queries_electron_number提前，为了综合多头注意力结果
        updated_queries = updated_queries.permute(0, 2, 1, 3)
        # 为了使得在内存上是连续的
        updated_queries = updated_queries.contiguous()
        # 变为输出维度的样子
        updated_queries = updated_queries.view(batch_size, queries_electron_number,
                                               self.head_numbers * self.queries_dimension)
        # 经过全连接层
        updated_queries = self.fc_output(updated_queries)

        return updated_queries


class MultiHeadAttention(Module):
    '''
    Multi-head attention layer with Dropout and Layer Normalization.
    '''

    def __init__(self, d_model, d_k, d_v, h, dropout=.1, identity_map_reordering=False, can_be_stateful=False,
                 attention_module=None, attention_module_kwargs=None):
        super(MultiHeadAttention, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        if attention_module is not None:
            if attention_module_kwargs is not None:
                self.attention = attention_module(d_model=d_model, d_k=d_k, d_v=d_v, h=h, **attention_module_kwargs)
            else:
                self.attention = attention_module(d_model=d_model, d_k=d_k, d_v=d_v, h=h)
        else:
            self.attention = ScaledDotProductAttention(d_model=d_model, d_k=d_k, d_v=d_v, h=h)
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(d_model)

        self.can_be_stateful = can_be_stateful
        if self.can_be_stateful:
            self.register_state('running_keys', torch.zeros((0, d_model)))
            self.register_state('running_values', torch.zeros((0, d_model)))

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        if self.can_be_stateful and self._is_stateful:
            self.running_keys = torch.cat([self.running_keys, keys], 1)
            keys = self.running_keys

            self.running_values = torch.cat([self.running_values, values], 1)
            values = self.running_values

        if self.identity_map_reordering:
            q_norm = self.layer_norm(queries)
            k_norm = self.layer_norm(keys)
            v_norm = self.layer_norm(values)
            out = self.attention(q_norm, k_norm, v_norm, attention_mask, attention_weights)
            out = queries + self.dropout(torch.relu(out))
        else:
            out = self.attention(queries, keys, values, attention_mask, attention_weights)
            out = self.dropout(out)
            out = self.layer_norm(queries + out)
        return out

