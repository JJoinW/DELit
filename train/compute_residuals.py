import pandas as pd
import numpy as np

# 加载数据
responses = pd.read_csv("llm_item_train/response_matrix.csv")
items = pd.read_csv("llm_item_train/item_parameters_full.csv")
students = pd.read_csv("llm_item_train/student_abilities_full.csv")

# 确保题目顺序正确
items = items.sort_values("item_id").reset_index(drop=True)

models = students['model_name'].tolist()
students = students.set_index("model_name")

predictions = pd.DataFrame({'id': items['item_id']})
residuals = pd.DataFrame({'id': items['item_id']})

# 题目的参数矩阵
a_matrix = items[['a_knowledge', 'a_retrieval', 'a_reasoning', 'a_summary']].values
b_vec = items['difficulty'].values
c_vec = items['guessing'].values

for model in models:
    # 获取该模型的能力值
    theta_m = students.loc[model, ['knowledge', 'retrieval', 'reasoning', 'summary']].values
    
    # 计算 logit = \sum(a * theta) - b
    logit = np.sum(a_matrix * theta_m, axis=1) - b_vec
    
    # 计算预测概率
    prob_base = 1.0 / (1.0 + np.exp(-logit))
    prob = c_vec + (1 - c_vec) * prob_base
    
    predictions[model] = prob
    
    # 计算残差 (真实值 - 预测值)
    if model in responses.columns:
        # 使用 responses 表中对应题目的真实值
        # 这里假设 responses 表也是按照 id 排序的，如果不保证就 match 一下
        true_resp_series = responses.set_index('id')[model]
        
        # 为了防错，通过映射保证对齐
        true_resp = items['item_id'].map(true_resp_series).values
        
        res = true_resp - prob
        residuals[model] = res
    else:
        residuals[model] = np.nan

predictions.to_csv("llm_item_train/predictions_matrix.csv", index=False)
residuals.to_csv("llm_item_train/residuals_matrix.csv", index=False)

print("保存成功：")
print("预测率矩阵: llm_item_train/predictions_matrix.csv")
print("残差矩阵: llm_item_train/residuals_matrix.csv")
