import pandas as pd
import argparse
import os

def evaluate_subset(exclude_models=None):
    if exclude_models is None:
        exclude_models = []
        
    response_file = "llm_item_train/response_matrix.csv"
    mapping_file = "llm_item_train/model_id_mapping.csv"
    
    if not os.path.exists(response_file) or not os.path.exists(mapping_file):
        print(f"找不到必要的数据文件: {response_file} 或 {mapping_file}")
        return
        
    df = pd.read_csv(response_file)
    mapping_df = pd.read_csv(mapping_file)
    
    # 构建模型ID到名称的映射，方便展示
    model_mapping = dict(zip(mapping_df['model_id'], mapping_df['model_name']))
    
    # 所有的模型列
    all_models = [col for col in df.columns if col.startswith('model_')]
    
    # 将输入的排除项转换为形如 'model_1', 'model_3' 等格式
    exclude_cols = []
    for m in exclude_models:
        # 如果传入的是纯数字
        if isinstance(m, int) or str(m).isdigit():
            exclude_cols.append(f"model_{m}")
        # 如果传入的是形如 'model_1' 的字符串
        elif m in all_models:
            exclude_cols.append(m)
        else:
            print(f"警告: 无法识别的模型标识符 '{m}'，将被忽略")
            
    # 计算剩余实际参与测试的模型集合
    remaining_models = [m for m in all_models if m not in exclude_cols]
    
    print("=" * 60)
    print("【排除的强力模型】:")
    for m in exclude_cols:
        print(f"  - {m}: {model_mapping.get(m, 'Unknown')}")
        
    print("\n【剩余参与评估的模型】:")
    for m in remaining_models:
        print(f"  - {m}: {model_mapping.get(m, 'Unknown')}")
    print("=" * 60)
    
    if not remaining_models:
        print("错误: 所有模型均被排除，没有剩余模型可以评估。")
        return
        
    # 分析计算
    total_questions = len(df)
    
    # 核心逻辑：只要剩余模型中有一个做对(值为1)，这道题就算作对
    # 对所有剩余模型列，横向求最大值
    df['subset_correct'] = df[remaining_models].max(axis=1)
    
    correct_count = df['subset_correct'].sum()
    accuracy = correct_count / total_questions if total_questions > 0 else 0
    
    print(f"\n统计结果:")
    print(f"-> 总题目数: {total_questions}")
    print(f"-> 剩余模型群体【至少一个做对】的题目数量: {int(correct_count)}")
    print(f"-> 剩余模型联合正确率: {accuracy:.2%}")
    print("=" * 60)

if __name__ == "__main__":
    # 使用 argparse 方便进行命令行调用探索
    parser = argparse.ArgumentParser(description="评估去除某些强力模型后，剩余弱模型的联合答题能力（只要有一个答对即算对）")
    parser.add_argument(
        '-e', '--exclude', 
        nargs='*', 
        default=[1, 2, 5, 3, 4, 6, 7, 9], 
        help="要排除的模型编号（如 1 3 4）或名称（如 model_1）。默认排除: 1 3 4 6 7"
    )
    
    args = parser.parse_args()
    evaluate_subset(args.exclude)
