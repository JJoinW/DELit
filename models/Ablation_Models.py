import torch
import torch.nn as nn

class Standard_MIRT(nn.Module):
    def __init__(self, num_students, num_items, q_matrix):
        """
        标准 MIRT 模型：去除神经网络结构，直接学习 student 和 item 的嵌入向量（即各个自由参数）。
        为兼容基于 one-hot 编码的 DataLoader，我们使用无偏置的 Linear 层来实现 Embedding 的等效效果。
        """
        super().__init__()
        self.register_buffer('q_matrix', q_matrix.float())

        # 直接作为自由参数矩阵学习，等效于 nn.Embedding
        self.student_emb = nn.Linear(num_students, 4, bias=False)
        self.item_emb = nn.Linear(num_items, 5, bias=False)

    def forward(self, student, item):
        # student: [batch, num_students] (one-hot)
        # item: [batch, num_items] (one-hot)
        theta = self.student_emb(student)
        
        item_params = self.item_emb(item)
        raw_a = item_params[:, :4]
        b = item_params[:, 4].unsqueeze(1)

        q_mask = torch.matmul(item, self.q_matrix)
        a = nn.functional.softplus(raw_a) * q_mask

        logit = (a * theta).sum(dim=-1, keepdim=True) - b
        return torch.sigmoid(logit)

class Neural_UIRT(nn.Module):
    def __init__(self, num_students, num_items, hidden_dim=64):
        """
        单一能力 Neural IRT (单维模型)：不考虑 Q-matrix，将所有题目视为考察单一能力。
        保留神经网络特征提取，但 student 输出 1 维能力，item 输出 1 维区分度和 1 维难度。
        """
        super().__init__()
        self.student_net = nn.Sequential(
            nn.Linear(num_students, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.item_net = nn.Sequential(
            nn.Linear(num_items, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, student, item):
        theta = self.student_net(student) # [batch, 1]
        
        item_params = self.item_net(item) # [batch, 2]
        raw_a = item_params[:, 0].unsqueeze(1) # [batch, 1]
        b = item_params[:, 1].unsqueeze(1)     # [batch, 1]

        # 无 Q-matrix 约束，单维区分度
        a = nn.functional.softplus(raw_a)
        
        logit = a * theta - b
        return torch.sigmoid(logit)