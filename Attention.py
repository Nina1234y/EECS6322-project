import torch


class PointWiseFeedForward(torch.nn.Module):

    def __init__(self, hidden_units, dropout_rate):
        super(PointWiseFeedForward, self).__init__()
        self.conv1d_1 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout_1 = torch.nn.Dropout(p=dropout_rate)
        self.relu = torch.nn.ReLU()
        self.conv1d_2 = torch.nn.Conv1d(hidden_units, hidden_units, kernel_size=1)
        self.dropout_2 = torch.nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x1 = self.conv1d_1(x.transpose(-1, -2))
        x2 = self.dropout_1(x1)
        x3 = self.relu(x2)
        x4 = self.conv1d_2(x3)
        x5 = self.dropout_2(x4)
        x6 = x5.transpose(-1, -2)
        x6 += x
        return x6


class TimeAwareMultiHeadAttention(torch.nn.Module):

    def __init__(self, hidden_units, heads, dropout_rate, device):
        super(TimeAwareMultiHeadAttention, self).__init__()
        self.hidden_units = hidden_units
        self.Q = torch.nn.Linear(hidden_units, hidden_units)
        self.K = torch.nn.Linear(hidden_units, hidden_units)
        self.V = torch.nn.Linear(hidden_units, hidden_units)
        self.dropout_rate = dropout_rate
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.heads = heads
        self.head_size = hidden_units // heads
        self.softmax = torch.nn.Softmax(dim=-1)
        self.device = device

    def forward(self, query, key, time_mask, attention_mask, time_matrix_Key, time_matrix_Value, distant_matrix_Key,
                distant_matrix_Value, abs_pos_Key, abs_pos_Value):
        Query, Key, Value = self.Q(query), self.K(key), self.V(key)

        Qy = torch.cat(torch.split(Query, self.head_size, dim=2), dim=0)
        Ky = torch.cat(torch.split(Key, self.head_size, dim=2), dim=0)
        Ve = torch.cat(torch.split(Value, self.head_size, dim=2), dim=0)
        K_timematrix = torch.cat(torch.split(time_matrix_Key, self.head_size, dim=3), dim=0)
        V_timematrix = torch.cat(torch.split(time_matrix_Value, self.head_size, dim=3), dim=0)
        K_distantmatrix = torch.cat(torch.split(distant_matrix_Key, self.head_size, dim=3), dim=0)
        V_distantmatrix = torch.cat(torch.split(distant_matrix_Value, self.head_size, dim=3), dim=0)
        abs_pos_K = torch.cat(torch.split(abs_pos_Key, self.head_size, dim=2), dim=0)
        abs_pos_V = torch.cat(torch.split(abs_pos_Value, self.head_size, dim=2), dim=0)

        attention_weights = Qy.matmul(torch.transpose(Ky, 1, 2))
        attention_weights += Qy.matmul(torch.transpose(abs_pos_K, 1, 2))
        attention_weights += K_timematrix.matmul(Qy.unsqueeze(-1)).squeeze(-1)
        attention_weights += K_distantmatrix.matmul(Qy.unsqueeze(-1)).squeeze(-1)
        attention_weights = attention_weights / (Ky.shape[-1] ** 0.5)

        time_mask = time_mask.unsqueeze(-1).repeat(self.heads, 1, 1)
        time_mask = time_mask.expand(-1, -1, attention_weights.shape[-1])
        attention_mask = attention_mask.unsqueeze(0).expand(attention_weights.shape[0], -1, -1)
        paddings = torch.ones(attention_weights.shape) * (-2 ** 32 + 1)
        paddings = paddings.to(self.device)
        attention_weights = torch.where(time_mask, paddings, attention_weights)
        attention_weights = torch.where(attention_mask, paddings, attention_weights)
        attention_weights = self.softmax(attention_weights)
        attention_weights = self.dropout(attention_weights)

        outputs = attention_weights.matmul(Ve)
        outputs += attention_weights.matmul(abs_pos_V)
        outputs += attention_weights.unsqueeze(2).matmul(V_timematrix).reshape(outputs.shape).squeeze(2)
        outputs += attention_weights.unsqueeze(2).matmul(V_distantmatrix).reshape(outputs.shape).squeeze(2)
        outputs = torch.cat(torch.split(outputs, Query.shape[0], dim=0), dim=2)

        return outputs

