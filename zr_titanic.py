import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score # 引入 cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
# from sklearn.linear_model import LogisticRegression

# 1. 数据加载
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    print("数据加载成功！")
except FileNotFoundError:
    print("错误：请确保 train.csv 和 test.csv 文件与您的 Python 脚本在同一目录下。")
    exit()

# 保存测试集的 PassengerId
test_passenger_ids = test_df['PassengerId']

# 将训练集和测试集合并，方便统一进行特征工程
combined_df = pd.concat([train_df.drop('Survived', axis=1), test_df], ignore_index=True)


# --- 2. 特征工程 ---
print("\n开始进行特征工程...")

# 提取 Title
def extract_title(name):
    title_search = re.search(' ([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""

combined_df['Title'] = combined_df['Name'].apply(extract_title)
# 更精细的 Title 映射
combined_df['Title'] = combined_df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona', 'Mme'], 'Rare')
combined_df['Title'] = combined_df['Title'].replace('Mlle', 'Miss')
combined_df['Title'] = combined_df['Title'].replace('Ms', 'Miss')


# 计算 FamilySize 和 IsAlone
combined_df['FamilySize'] = combined_df['SibSp'] + combined_df['Parch'] + 1
combined_df['IsAlone'] = (combined_df['FamilySize'] == 1).astype(int)


# 处理 Cabin
def process_cabin(cabin):
    if pd.isna(cabin):
        return 'Missing'
    else:
        return cabin[0] # 提取第一个字母作为甲板

combined_df['CabinDeck'] = combined_df['Cabin'].apply(process_cabin)
# 可以进一步将稀有甲板合并，例如 T 通常和 A 相似或并入 Rare
# combined_df['CabinDeck'] = combined_df['CabinDeck'].replace('T', 'A') # 示例合并

# 提取 Ticket 组大小
ticket_counts = combined_df['Ticket'].value_counts()
combined_df['TicketGroupSize'] = combined_df['Ticket'].map(ticket_counts)
combined_df['TicketGroupSize_Category'] = pd.cut(combined_df['TicketGroupSize'],
                                                  bins=[0, 1, 4, np.inf], # 1人, 2-4人, 5人以上
                                                  labels=['Alone', 'SmallGroup', 'LargeGroup'],
                                                  right=True).astype(str)
combined_df['TicketGroupSize_Category'] = combined_df['TicketGroupSize_Category'].replace('nan', 'Alone')


# 基于 Title/Pclass/Sex 填充 Age 缺失值
grouped_age = combined_df.groupby(['Title', 'Pclass', 'Sex'])['Age'].median()
def fill_age(row):
    if pd.isna(row['Age']):
        try:
            return grouped_age[row['Title'], row['Pclass'], row['Sex']]
        except KeyError:
            try:
                 return combined_df.groupby(['Title', 'Pclass'])['Age'].median()[row['Title'], row['Pclass']]
            except KeyError:
                 return combined_df['Age'].median()
    else:
        return row['Age']

combined_df['Age'] = combined_df.apply(fill_age, axis=1)


# 处理 Fare 的缺失值（测试集有1个缺失值）
combined_df['Fare'] = combined_df['Fare'].fillna(combined_df['Fare'].median())

# 对 Fare 进行 Log 变换
# 使用 FunctionTransformer 封装 Log 变换，可以放入 pipeline
log1p_transformer = FunctionTransformer(np.log1p, validate=False)


# 对 Fare 进行分箱 (保留作为额外的分类特征)
combined_df['FareBin'] = pd.qcut(combined_df['Fare'], 4, labels=['1st', '2nd', '3rd', '4th']).astype(str)
combined_df['FareBin'] = combined_df['FareBin'].replace('nan', 'Unknown')


print("特征工程完成！")


# 重新分离训练集和测试集
X_train = combined_df.iloc[:len(train_df)].copy()
X_test = combined_df.iloc[len(train_df):].copy()
y_train = train_df['Survived']


# --- 3. 更新特征列表和预处理 ---
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Embarked',
            'Title', 'FamilySize', 'IsAlone', 'CabinDeck', 'TicketGroupSize', 'TicketGroupSize_Category', 'Fare', 'FareBin']
# 注意：这里我们保留了原始的 'Fare'，Log 变换将在 Pipeline 的预处理步骤中进行

# 选择需要使用的特征列
X_train = X_train[features]
X_test = X_test[features]

# 确定处理后的数值型和分类型特征
# Age 已手动填充
# Fare 将在 Pipeline 中进行 Log 变换和标准化
numerical_features = ['Age', 'SibSp', 'Parch', 'FamilySize', 'TicketGroupSize', 'Fare'] # 注意这里包含原始 'Fare'
categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'IsAlone', 'CabinDeck', 'FareBin', 'TicketGroupSize_Category']


# 创建预处理的 Pipeline
# 对于数值型特征：填充缺失值（理论上Age已填充，Fare已填充，主要防范未来数据问题），Log 变换，然后进行标准化
# 对于分类型特征：填充缺失值（使用最多的值），然后进行独热编码
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')), # 对数值列进行中位数填充，防止未知缺失
            ('log_transform', log1p_transformer),          # Log 变换 Fare (以及其他数值列，但主要影响Fare)
            ('scaler', StandardScaler())]), numerical_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')), # 填充分类列缺失值 (Embarked等)
            ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ],
    remainder='passthrough' # 保留未处理的列（这里应该没有）
)


# --- 4. 模型选择 (使用 XGBoost) ---
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=0.1
)


# --- 5. 构建完整的机器学习 Pipeline：预处理 -> 模型 ---
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', model)])

# --- 6. 进行五折交叉验证评估 ---
print("\n开始进行五折分层交叉验证评估...")

# 使用 StratifiedKFold 确保每个折的 Survived 比例一致
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 使用 cross_val_score 计算交叉验证得分
# pipeline 会在每次交叉验证迭代时，在训练折上进行预处理和训练，并在验证折上进行评估
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=skf, scoring='accuracy', n_jobs=-1) # n_jobs=-1 使用所有可用核心

print(f"交叉验证得分 (Accuracy): {cv_scores}")
print(f"交叉验证平均得分: {cv_scores.mean():.4f}")
print(f"交叉验证得分标准差: {cv_scores.std():.4f}")

print("交叉验证评估完成！")


# --- 7. 使用整个训练集训练最终模型 ---
print("\n使用整个训练集训练最终模型...")
pipeline.fit(X_train, y_train)
print("最终模型训练完成！")


# --- 8. 模型预测 ---
print("开始进行预测...")
predictions = pipeline.predict(X_test)
print("预测完成！")

# --- 9. 生成提交文件 ---
submission_df = pd.DataFrame({'PassengerId': test_passenger_ids, 'Survived': predictions})
submission_df['Survived'] = submission_df['Survived'].astype(int)
submission_df.to_csv('submission.csv', index=False)

print("\n提交文件 'submission.csv' 已生成成功！")
print("文件预览:")
print(submission_df.head())