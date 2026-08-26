import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

# 复用 train_mirt 里的工具函数
from llm_item_train.train_mirt import MIRTDataset, evaluate
from models.Ablation_Models import Standard_MIRT, Neural_UIRT

def train_and_eval(model, model_name, train_loader, val_loader, test_loader, device):
    print(f"\n{'='*50}\n开始训练消融模型: {model_name}\n{'='*50}")
    optimizer = optim.Adam(model.parameters(), lr=0.003, weight_decay=0.0001)
    criterion = nn.BCELoss()

    best_auc = 0.0
    patience = 5
    epochs_no_improve = 0

    for epoch in range(50):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
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

        if val_metrics['auc'] > best_auc:
            best_auc = val_metrics['auc']
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"llm_item_train/best_{model_name}.pth")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict(torch.load(f"llm_item_train/best_{model_name}.pth", weights_only=True))
    test_metrics, _, _, _ = evaluate(model, test_loader, device)

    print(f"[{model_name}] 最终评估结果:")
    print(f"ACC: {test_metrics['acc']:.4f}, F1: {test_metrics['f1']:.4f}, AUC: {test_metrics['auc']:.4f}")
    return test_metrics

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据加载
    q_matrix_df = pd.read_csv("llm_item_train/q_matrix.csv")
    response_df = pd.read_csv("llm_item_train/response_matrix.csv")

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
    
    # 划分数据集
    total_len = len(data_triples)
    train_size = int(0.8 * total_len)
    val_size = int(0.1 * total_len)
    test_size = total_len - train_size - val_size

    torch.manual_seed(42)
    train_data, val_data, test_data = random_split(data_triples, [train_size, val_size, test_size])

    train_loader = DataLoader(MIRTDataset(train_data, num_students, num_items), batch_size=512, shuffle=True)
    val_loader = DataLoader(MIRTDataset(val_data, num_students, num_items), batch_size=512, shuffle=False)
    test_loader = DataLoader(MIRTDataset(test_data, num_students, num_items), batch_size=512, shuffle=False)

    # 1. 运行消融模型1：去掉神经网络，采用传统 Embedding 标准 MIRT
    model_std_mirt = Standard_MIRT(num_students, num_items, q_matrix).to(device)
    train_and_eval(model_std_mirt, "Standard_MIRT", train_loader, val_loader, test_loader, device)

    # 2. 运行消融模型2：不考虑 Q 矩阵，采用单一能力单维度的 Neural UIRT
    model_neural_uirt = Neural_UIRT(num_students, num_items, hidden_dim=64).to(device)
    train_and_eval(model_neural_uirt, "Neural_UIRT", train_loader, val_loader, test_loader, device)

if __name__ == "__main__":
    main()