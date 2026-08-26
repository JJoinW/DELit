import torch
import torch.nn as nn

class NonComp_Neural_MIRT(nn.Module):
    def __init__(self, num_students, num_items, q_matrix, hidden_dim=64, model_type='2PL'):
        """
        非补偿型 (Non-compensatory) Neural MIRT
        :param num_students: 被试数量 (大模型数量)
        :param num_items: 题目数量
        :param q_matrix: [num_items, 4] 的 Q-matrix Tensor
        :param hidden_dim: 隐层维度
        :param model_type: 模式选项
                           '1PL' 区分度固定
                           '2PL' 学习多维区分度和维度难度
                           '3PL' 加上猜测参数 (guessing)
        """
        super().__init__()
        self.model_type = model_type
        
        # 1. 注册 Q-matrix 为不参与梯度更新的 buffer
        self.register_buffer('q_matrix', q_matrix.float())

        # 2. Student 网络，输出 4 维能力向量: [knowledge, retrieval, reasoning, summary]
        self.student_net = nn.Sequential(
            nn.Linear(num_students, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4)
        )
        
        # 3. Item 网络
        # 在非补偿型模型中，针对每个维度都需要一个独立的难度参数 b，因此难度 b 应该是 4 维。
        # 1PL: 学习难度 b (4维)。 (区分度由 q_matrix 决定)
        # 2PL: 区分度 a (4维) + 难度 b (4维) = 8维
        # 3PL: 区分度 a (4维) + 难度 b (4维) + 猜测参数 c (1维) = 9维
        if model_type == '1PL':
            item_out_dim = 4
        elif model_type == '2PL':
            item_out_dim = 8
        elif model_type == '3PL':
            item_out_dim = 9
        else:
            raise ValueError("model_type必须为 '1PL', '2PL' 或 '3PL'")
            
        self.item_net = nn.Sequential(
            nn.Linear(num_items, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, item_out_dim)
        )

    def forward(self, student, item):
        """
        :param student: [batch_size, num_students]
        :param item: [batch_size, num_items]
        :return: prob [batch_size, 1]
        """
        theta = self.student_net(student)  # shape: [batch_size, 4]
        item_params = self.item_net(item)
        q_mask = torch.matmul(item, self.q_matrix)  # shape: [batch_size, 4]

        if self.model_type == '1PL':
            a = q_mask
            b = item_params[:, :4] * q_mask  # 对难度参数b进行 q 矩阵掩码
            c = torch.zeros(b.shape[0], 1, device=b.device)
        elif self.model_type == '2PL':
            raw_a = item_params[:, :4]
            b = item_params[:, 4:8] * q_mask
            a = nn.functional.softplus(raw_a) * q_mask
            c = torch.zeros(b.shape[0], 1, device=b.device)
        elif self.model_type == '3PL':
            raw_a = item_params[:, :4]
            b = item_params[:, 4:8] * q_mask
            c_raw = item_params[:, 8].unsqueeze(1)
            
            a = nn.functional.softplus(raw_a) * q_mask
            c = torch.sigmoid(c_raw)

        # 非补偿型 (Non-compensatory) MIRT 数学结构
        # 针对每个所需知识点计算独立的达成概率，然后进行连乘
        prob_k = torch.sigmoid(a * theta + b)  # shape: [batch_size, 4]
        
        # 只考虑在 q 矩阵中有对应技能的概率值，无关维度概率主动置为 1 以进行无损连乘
        prob_k_masked = torch.where(q_mask > 0, prob_k, torch.ones_like(prob_k))
        
        prob_base = torch.prod(prob_k_masked, dim=-1, keepdim=True)  # shape: [batch_size, 1]
        
        # 加上猜测参数（Lower asymptote）
        prob = c + (1 - c) * prob_base
        
        return prob
