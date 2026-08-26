import sys
import os
# 添加项目根目录到 Python 路径中，以便能够顺利导入 models 下的模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
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
def evaluate(model, dataloader, device):
    model.eval()
    all_probs, all_preds, all_labels = [], [], []
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

            y_pred_binary = (y_pred_probs > 0.5).float()

            all_probs.extend(y_pred_probs.cpu().numpy())
            all_preds.extend(y_pred_binary.cpu().numpy())
            all_labels.extend(y_true.numpy())

    try:
        auc_score = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc_score = 0.5

    metrics = {
        'acc': accuracy_score(all_labels, all_preds),
        'auc': auc_score,
        'f1': f1_score(all_labels, all_preds, average='binary', zero_division=0)
    }

    return metrics, all_labels, all_preds, all_probs


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

    # 从 q_matrix.csv 中提取 Q矩阵, 对应能力为 a,b,c,d
    # 假设 Q矩阵 文件有 [id, a, b, c, d] 等列
    q_matrix = torch.tensor(q_matrix_df[['a', 'b', 'c', 'd']].values, dtype=torch.float32)
    num_items = len(q_matrix_df)
    
    # 从 response_matrix.csv 中提取数据三元组
    # 假设列中有以 "model_" 开头标注的不同大模型被试
    student_cols = [col for col in response_df.columns if col.startswith('model')]
    num_students = len(student_cols)
    
    data_triples = []
    for qid in range(num_items):
        for sid, stu_col in enumerate(student_cols):
            label = response_df.iloc[qid][stu_col]
            if not pd.isna(label):
                data_triples.append((sid, qid, int(label)))
    
    print(f"加载作答记录数量: {len(data_triples)} (num_students: {num_students}, num_items: {num_items})")

    # 2. 8:1:1 数据随机划分
    total_len = len(data_triples)
    train_size = int(0.8 * total_len)
    val_size = int(0.1 * total_len)
    test_size = total_len - train_size - val_size

    # 固定随机种子确保结果可复现(按需)
    torch.manual_seed(42)
    train_data, val_data, test_data = random_split(data_triples, [train_size, val_size, test_size])

    train_dataset = MIRTDataset(train_data, num_students, num_items)
    val_dataset = MIRTDataset(val_data, num_students, num_items)
    test_dataset = MIRTDataset(test_data, num_students, num_items)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    print(f"数据划分完成 - 训练集：{len(train_dataset)} | 验证集：{len(val_dataset)} | 测试集：{len(test_dataset)}")

    MODEL_TYPE_TO_TRAIN = '2PL'  # 请在这里指定训练的模式 ('1PL' 或 '2PL' 或 '3PL')
    USE_NONCOMPENSATORY = False  # 是否使用非补偿型 MIRT 模型

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

    best_auc = 0.0
    patience = 8
    epochs_no_improve = 0

    # 4. 训练循环
    for epoch in range(50):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}", unit="batch", leave=False):
            optimizer.zero_grad()
            student = batch["student"].to(device)
            item = batch["item"].to(device)
            labels = batch["label"].to(device)
            
            probs = model(student, item)
            loss = criterion(probs.squeeze(), labels)
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_metrics, _, _, _ = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1} | Train Loss: {avg_loss:.4f} | Val ACC: {val_metrics['acc']:.4f} | Val F1: {val_metrics['f1']:.4f} | Val AUC: {val_metrics['auc']:.4f}")

        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            epochs_no_improve = 0
            torch.save(model.state_dict(), "llm_item_train/best_model_neural_mirt.pth")
            print(f"-> 模型已更新，最佳 Val AUC: {best_auc:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered.")
                break

    # 5. 测试评估
    model.load_state_dict(torch.load("llm_item_train/best_model_neural_mirt.pth", weights_only=True))
    print("\n加载最佳模型进行最终测试评估...")
    test_metrics, test_labels, test_preds, test_probs = evaluate(model, test_loader, device)

    print(f"Final Test Performance:")
    print(f"ACC: {test_metrics['acc']:.4f}, F1: {test_metrics['f1']:.4f}, AUC: {test_metrics['auc']:.4f}")

    # 保存测试集的预测结果（可选）
    pd.DataFrame({
        'ground_truth': test_labels,
        'prediction': test_preds,
        'probability': test_probs
    }).to_csv("llm_item_train/test_predictions_neural_mirt.csv", index=False)

    # 6. 参数导出
    print("\n导出参数: Student Abilities & Item Parameters...")
    model.eval()
    
    # 导出 LLM / Student 的 4维能力
    student_ability_list = []
    with torch.no_grad():
        for sid in range(num_students):
            student_vec = torch.zeros(num_students).float()
            student_vec[sid] = 1.0
            
            # shape [1, 4]
            theta = model.student_net(student_vec.unsqueeze(0).to(device)).squeeze(0).cpu().numpy()
            
            student_ability_list.append({
                "model_name": student_cols[sid],
                "knowledge": theta[0],
                "retrieval": theta[1],
                "reasoning": theta[2],
                "summary": theta[3]
            })
    pd.DataFrame(student_ability_list).to_csv("llm_item_train/student_abilities.csv", index=False)

    # 导出 Item params (受 Q-matrix 约束计算后)
    item_params_list = []
    with torch.no_grad():
        for qid in range(num_items):
            # 构造 one-hot item 向量以通过 item_net 获取原始参数
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
                "item_id": qid + 1,        # 采用1从开始索引
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
    pd.DataFrame(item_params_list).to_csv("llm_item_train/item_parameters.csv", index=False)

    print("所有的参数导出已完成：llm_item_train/student_abilities.csv, llm_item_train/item_parameters.csv")

if __name__ == "__main__":
    main()