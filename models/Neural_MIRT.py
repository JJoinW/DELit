import torch
import torch.nn as nn

class Neural_MIRT(nn.Module):
    def __init__(self, num_students, num_items, q_matrix, hidden_dim=64, model_type='2PL'):
        """
        :param num_students: 被试数量 (大模型数量)
        :param num_items: 题目数量
        :param q_matrix: [num_items, 4] 的 Q-matrix Tensor
        :param hidden_dim: 隐层维度
        :param model_type: 模式选项
                           '1PL' 表示只学习难度 (用固定 Q-matrix 作为区分度)
                           '2PL' 学习难度与多维区分度
                           '3PL' 学习难度、多维区分度以及猜测参数 (guessing)
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
        # 1PL 只需要输出难度 b (1维)。
        # 2PL 输出区分度 a (4维) + 难度 b (1维) = 5维
        # 3PL 输出区分度 a (4维) + 难度 b (1维) + 猜测参数 c (1维) = 6维
        if model_type == '1PL':
            item_out_dim = 1
        elif model_type == '2PL':
            item_out_dim = 5
        elif model_type == '3PL':
            item_out_dim = 6
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
        # 提取 4 维能力值
        theta = self.student_net(student)  # shape: [batch_size, 4]
        
        # 提取题目参数
        item_params = self.item_net(item)
        
        # 提取基于当前批次题目的 Q-mask
        # item 是 one-hot 编码，与 q_matrix 矩阵相乘可提取属于对应题目的 q_mask
        q_mask = torch.matmul(item, self.q_matrix)  # shape: [batch_size, 4]

        if self.model_type == '1PL':
            # 1PL 模式下：题目区分度特征为恒定值 (即 q_mask 本身)，仅学习难度值 b
            a = q_mask
            b = item_params[:, 0].unsqueeze(1)
            c = torch.zeros_like(b)  # 猜测率为0
        elif self.model_type == '2PL':
            # 2PL 模式下：同时学习 raw_a 和难度 b
            raw_a = item_params[:, :4]         # shape: [batch_size, 4]
            b = item_params[:, 4].unsqueeze(1) # shape: [batch_size, 1]
            # 计算区分度 a，要求：使用 softplus 保证非负，并实施 Q-matrix hard 掩码
            a = nn.functional.softplus(raw_a) * q_mask  # shape: [batch_size, 4]
            c = torch.zeros_like(b)  # 猜测率为0
        elif self.model_type == '3PL':
            raw_a = item_params[:, :4]
            b = item_params[:, 4].unsqueeze(1)
            c_raw = item_params[:, 5].unsqueeze(1)
            
            a = nn.functional.softplus(raw_a) * q_mask
            c = torch.sigmoid(c_raw)  # 使用 sigmoid 保证猜测截距在 (0, 1) 区间内

        # 多维 compensatory MIRT 数学结构
        logit = (a * theta).sum(dim=-1, keepdim=True) - b  # shape: [batch_size, 1]
        
        # sigmoid 映射为基础答对概率
        prob_base = torch.sigmoid(logit)  # shape: [batch_size, 1]
        
        # 加上猜测参数（Lower asymptote）
        prob = c + (1 - c) * prob_base
        
        return prob
