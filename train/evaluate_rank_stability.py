import sys
import os
# 添加项目根目录到 Python 路径中
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from scipy.stats import kendalltau
import numpy as np
from tqdm import tqdm

from models.Neural_MIRT import Neural_MIRT
from models.NonComp_Neural_MIRT import NonComp_Neural_MIRT

# -------------------- Dataset Class --------------------
class MIRTDataset(Dataset):
    def __init__(self, data, num_students, num_items):
        self.data = data
        self.num_students = num_students
        self.num_items = num_items

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sid, qid, label = self.data[idx]
        student_vec = torch.zeros(self.num_students)
        student_vec[sid] = 1.0
        item_vec = torch.zeros(self.num_items)
        item_vec[qid] = 1.0
        return {
            "student": student_vec.float(),
            "item": item_vec.float(),
            "label": torch.tensor(label, dtype=torch.float),
        }

# -------------------- Evaluation Function --------------------
def evaluate_model(model, dataloader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            student = batch["student"].to(device)
            item = batch["item"].to(device)
            y_true = batch["label"]

            y_pred_probs = model(student, item)

            if y_pred_probs.dim() > 1:
                y_pred_probs = y_pred_probs.squeeze()
            if y_pred_probs.dim() == 0:
                y_pred_probs = y_pred_probs.unsqueeze(0)

            all_probs.extend(y_pred_probs.cpu().numpy())
            all_labels.extend(y_true.numpy())

    try:
        auc_score = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc_score = 0.5

    return auc_score

# -------------------- Training Routine --------------------
def train_and_get_abilities(data_subset, num_students, num_items, q_matrix, device, ModelClass, model_type):
    # 将此半部分数据再划分为 90% 训练和 10% 验证 (为了能够有 Early Stopping 机制)
    train_size = int(0.9 * len(data_subset))
    val_size = len(data_subset) - train_size
    train_data, val_data = random_split(data_subset, [train_size, val_size])

    train_loader = DataLoader(MIRTDataset(train_data, num_students, num_items), batch_size=512, shuffle=True)
    val_loader = DataLoader(MIRTDataset(val_data, num_students, num_items), batch_size=512, shuffle=False)

    model = ModelClass(
        num_students=num_students,
        num_items=num_items,
        q_matrix=q_matrix,
        hidden_dim=64,
        model_type=model_type
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=0.0001)
    criterion = nn.BCELoss()

    best_auc = 0.0
    patience = 8
    epochs_no_improve = 0
    best_model_state = None

    for epoch in range(40):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            student = batch["student"].to(device)
            item = batch["item"].to(device)
            labels = batch["label"].to(device)
            
            probs = model(student, item)
            loss = criterion(probs.squeeze(), labels)
            
            loss.backward()
            optimizer.step()

        val_auc = evaluate_model(model, val_loader, device)

        if val_auc > best_auc:
            best_auc = val_auc
            epochs_no_improve = 0
            # 仅保存在内存中
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    # 加载表现最好的 epoch 权重
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.to(device)
    model.eval()
    
    # 提取被试能力向量
    abilities = []
    with torch.no_grad():
        for sid in range(num_students):
            student_vec = torch.zeros(num_students).float()
            student_vec[sid] = 1.0
            theta = model.student_net(student_vec.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
            abilities.append(theta)
            
    return np.array(abilities) # shape: [num_students, 4]


# -------------------- Main Program --------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. 加载数据
    q_matrix_path = "llm_item_train/q_matrix.csv"
    response_matrix_path = "llm_item_train/response_matrix.csv"

    try:
        q_matrix_df = pd.read_csv(q_matrix_path)
        response_df = pd.read_csv(response_matrix_path)
    except FileNotFoundError as e:
        print(f"配置文件读取失败: {e.filename}。请检查路径。")
        return

    q_matrix = torch.tensor(q_matrix_df[['a', 'b', 'c', 'd']].values, dtype=torch.float32)
    num_items = len(q_matrix_df)
    
    student_cols = [col for col in response_df.columns if col.startswith('model')]
    num_students = len(student_cols)
    
    data_triples = []
    for qid in range(num_items):
        for sid, stu_col in enumerate(student_cols):
            label = response_df.iloc[qid][stu_col]
            if not pd.isna(label):
                data_triples.append((sid, qid, int(label)))
    
    print(f"全量作答记录数量: {len(data_triples)}")

    # 2. 对半划分数据 (Subset A 和 Subset B)
    torch.manual_seed(42)
    half_size = len(data_triples) // 2
    data_A, data_B = random_split(data_triples, [half_size, len(data_triples) - half_size])

    MODEL_TYPE = '2PL'         # ('1PL' 或 '2PL' 或 '3PL')
    USE_NONCOMPENSATORY = False  # 是否使用非补偿型 MIRT 模型
    ModelClass = NonComp_Neural_MIRT if USE_NONCOMPENSATORY else Neural_MIRT

    print(f"\n即将进行 Rank Stability 实验")
    print(f"模型模式: {MODEL_TYPE} | 非补偿型: {USE_NONCOMPENSATORY}")
    print(f"数据集已分成 Subset A ({len(data_A)}) 和 Subset B ({len(data_B)})")

    # 3. 分别在两部分数据上独立训练并获取能力参数
    print("\n--- 开始在 Subset A 上训练模型 ---")
    abilities_A = train_and_get_abilities(data_A, num_students, num_items, q_matrix, device, ModelClass, MODEL_TYPE)
    
    print("--- 开始在 Subset B 上训练模型 ---")
    abilities_B = train_and_get_abilities(data_B, num_students, num_items, q_matrix, device, ModelClass, MODEL_TYPE)

    # 4. 计算 Rank Stability (Kendall's Tau)
    dimensions = ["knowledge", "retrieval", "reasoning", "summary"]
    
    print("\n=== Rank Stability (Kendall's τ) Results ===")
    for i, dim in enumerate(dimensions):
        # 提取两组实验中学到的对应维度的能力值
        rank_A = abilities_A[:, i]
        rank_B = abilities_B[:, i]
        
        tau, p_value = kendalltau(rank_A, rank_B)
        print(f"Dimension [{dim.ljust(9)}]: Kendall's τ = {tau:.4f} \t (p-value = {p_value:.4e})")

if __name__ == "__main__":
    main()