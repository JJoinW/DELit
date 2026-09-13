import sys
import os
# 添加项目根目录到 Python 路径中，以便能够顺利导入 models 下的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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

    print(f"加载作答记录数量: {len(data_triples)} (num_students: {num_students}, num_items: {num_items})")

    # 2. 数据加载
    dataset = MIRTDataset(data_triples, num_students, num_items)
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

    MODEL_TYPE_TO_TRAIN = '2PL'  # 请在这里指定训练的模式 ('1PL' 或 '2PL' 或 '3PL')
    USE_NONCOMPENSATORY = False  # 是否使用非补偿型 MIRT 模型
    epochs = 7

    print(f"\n即将训练的 MIRT 模型模式: {MODEL_TYPE_TO_TRAIN} | 非补偿型: {USE_NONCOMPENSATORY}")

    # 3. 初始化 Neural MIRT 模型
    ModelClass = NonComp_Neural_MIRT if USE_NONCOMPENSATORY else Neural_MIRT
    model = ModelClass(
        num_students=num_students,
        num_items=num_items,
        q_matrix=q_matrix,
        hidden_dim=64,
        model_type=MODEL_TYPE_TO_TRAIN
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=0.0001)
    criterion = nn.BCELoss()

    # 4. 训练循环
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}", unit="batch", leave=False):
            optimizer.zero_grad()
            student = batch["student"].to(device)
            item = batch["item"].to(device)
            labels = batch["label"].to(device)

            probs = model(student, item)
            loss = criterion(probs.squeeze(), labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f}")

    # 5. 参数导出
    print("\n导出参数: Student Abilities & Item Parameters...")
    model.eval()

    # 导出 LLM / Student 的 4维能力
    student_ability_list = []
    with torch.no_grad():
        for sid in range(num_students):
            student_vec = torch.zeros(num_students).float()
            student_vec[sid] = 1.0

            theta = model.student_net(student_vec.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()

            student_ability_list.append({
                "model_name": student_cols[sid],
                "knowledge": theta[0],
                "retrieval": theta[1],
                "reasoning": theta[2],
                "summary": theta[3]
            })
    pd.DataFrame(student_ability_list).to_csv("llm_item_train/student_abilities_full.csv", index=False)

    # 导出 Item params (受 Q-matrix 约束计算后)
    item_params_list = []
    with torch.no_grad():
        for qid in range(num_items):
            item_vec = torch.zeros(num_items).float()
            item_vec[qid] = 1.0

            raw_params = model.item_net(item_vec.unsqueeze(0).to(device)).squeeze(0)
            is_noncomp = isinstance(model, NonComp_Neural_MIRT)

            if model.model_type == '1PL':
                if is_noncomp:
                    q_mask_item = q_matrix[qid].to(device)
                    b_vals = (raw_params[:4] * q_mask_item).cpu().numpy()
                else:
                    b_vals = [raw_params[0].item()]
                a_constrained = q_matrix[qid].cpu().numpy()
                c_val = 0.0
            elif model.model_type == '2PL':
                raw_a = raw_params[:4]
                q_mask_item = q_matrix[qid].to(device)
                if is_noncomp:
                    b_vals = (raw_params[4:8] * q_mask_item).cpu().numpy()
                else:
                    b_vals = [raw_params[4].item()]
                a_constrained = (nn.functional.softplus(raw_a) * q_mask_item).cpu().numpy()
                c_val = 0.0
            elif model.model_type == '3PL':
                raw_a = raw_params[:4]
                q_mask_item = q_matrix[qid].to(device)
                if is_noncomp:
                    b_vals = (raw_params[4:8] * q_mask_item).cpu().numpy()
                    c_val = torch.sigmoid(raw_params[8]).item()
                else:
                    b_vals = [raw_params[4].item()]
                    c_val = torch.sigmoid(raw_params[5]).item()

                a_constrained = (nn.functional.softplus(raw_a) * q_mask_item).cpu().numpy()

            record = {
                "item_id": qid + 1,
                "a_knowledge": float(a_constrained[0]),
                "a_retrieval": float(a_constrained[1]),
                "a_reasoning": float(a_constrained[2]),
                "a_summary": float(a_constrained[3]),
                "guessing": float(c_val)
            }
            if is_noncomp:
                record.update({
                    "b_knowledge": float(b_vals[0]),
                    "b_retrieval": float(b_vals[1]),
                    "b_reasoning": float(b_vals[2]),
                    "b_summary": float(b_vals[3]),
                })
            else:
                record["difficulty"] = float(b_vals[0])

            item_params_list.append(record)
    pd.DataFrame(item_params_list).to_csv("llm_item_train/item_parameters_full.csv", index=False)

    print("所有的参数导出已完成：llm_item_train/student_abilities_full.csv, llm_item_train/item_parameters_full.csv")

if __name__ == "__main__":
    main()