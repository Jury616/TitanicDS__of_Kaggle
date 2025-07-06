import pandas as pd
import numpy as np
import re # 用于正则表达式提取 Title
from sklearn.model_selection import train_test_split, GridSearchCV # 用于模型调优和交叉验证
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# 导入其他可能的模型
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from xgboost import XGBClassifier # 如果安装了 xgboost 可以取消注释
# from lightgbm import LGBMClassifier # 如果安装了 lightgbm 可以取消注释
# from sklearn.svm import SVC # 支持向量机

# 1. 数据加载
try:
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    # gender_submission_df = pd.read_csv('gender_submission.csv') # 用于参考提交格式
    print("数据加载成功！")
except FileNotFoundError:
    print("错误：请确保 train.csv, test.csv 文件与您的 Python 脚本在同一目录下。")
    exit()

# 保存测试集的 PassengerId，用于生成提交文件
test_passenger_ids = test_df['PassengerId']

# 为了方便统一进行特征工程，我们将训练集和测试集合并
# 在进行模型训练之前，我们会重新将它们分开
# 注意：在更复杂的场景或进行严格的交叉验证时，需要更小心地处理，避免数据泄露
combined_df = pd.concat([train_df.drop('Survived', axis=1), test_df], ignore_index=True)


# 2. 特征工程函数定义
def extract_title(name):
    """从姓名中提取称谓 (Title)"""
    title_search = re.search(' ([A-Za-z]+)\.', name)
    # 如果称谓存在，提取并返回
    if title_search:
        return title_search.group(1)
    return ""

def process_cabin(cabin):
    """处理 Cabin 特征，提取首字母或标记为缺失"""
    if pd.isna(cabin):
        return 'Missing'
    else:
        # 提取第一个字母作为甲板 (Deck)
        return cabin[0]

def bin_fare(fare):
    """对 Fare 进行分箱处理"""
    if pd.isna(fare):
        return 'Unknown' # 填充缺失值
    elif fare <= 7.91:
        return 'Low'
    elif fare <= 14.454:
        return 'Medium_Low'
    elif fare <= 31.0:
        return 'Medium_High'
    else:
        return 'High'

# 3. 应用特征工程
print("\n开始进行特征工程...")

# 提取 Title
combined_df['Title'] = combined_df['Name'].apply(extract_title)
# 对一些稀有称谓进行映射，合并到常见称谓中
# 常见的映射规则，可以根据数据分析结果调整
combined_df['Title'] = combined_df['Title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
combined_df['Title'] = combined_df['Title'].replace('Mlle', 'Miss')
combined_df['Title'] = combined_df['Title'].replace('Ms', 'Miss')
combined_df['Title'] = combined_df['Title'].replace('Mme', 'Mrs')


# 计算 FamilySize 和 IsAlone
combined_df['FamilySize'] = combined_df['SibSp'] + combined_df['Parch'] + 1
combined_df['IsAlone'] = 0
combined_df.loc[combined_df['FamilySize'] == 1, 'IsAlone'] = 1

# 处理 Cabin
combined_df['CabinDeck'] = combined_df['Cabin'].apply(process_cabin)

# 处理 Fare 的缺失值（在分箱前）
combined_df['Fare'] = combined_df['Fare'].fillna(combined_df['Fare'].median()) # 使用中位数填充 Fare 缺失值

# 对 Fare 进行分箱
combined_df['FareBin'] = combined_df['Fare'].apply(bin_fare)

print("特征工程完成！ 新增特征: Title, FamilySize, IsAlone, CabinDeck, FareBin")
# print(combined_df.head()) # 可以打印查看新的特征


# 重新分离训练集和测试集
X_train = combined_df.iloc[:len(train_df)].copy()
X_test = combined_df.iloc[len(train_df):].copy()
y_train = train_df['Survived'] # 目标变量只在原始训练集里


# 4. 更新特征列表和预处理
# 使用新的特征列表
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
            'Title', 'FamilySize', 'IsAlone', 'CabinDeck', 'FareBin']

# 选择需要使用的特征列
X_train = X_train[features]
X_test = X_test[features]


# 确定处理后的数值型和分类型特征
# 注意：这里的 'Fare' 仍然保留，但在预处理中可以进一步处理，或者依赖模型自己处理
# FamilySize 和 IsAlone 也是数值型
numerical_features = ['Age', 'SibSp', 'Parch', 'FamilySize', 'Fare']
# 新增的分类特征：Title, CabinDeck, FareBin
categorical_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'IsAlone', 'CabinDeck', 'FareBin'] # IsAlone 也可以看作分类


# 创建预处理的 Pipeline
# 对于数值型特征：填充缺失值（使用均值），然后进行标准化
# 对于分类型特征：填充缺失值（使用最多的值），然后进行独热编码
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')), # 填充 Age 和 Fare (SibSp, Parch, FamilySize 应该没有缺失)
            ('scaler', StandardScaler())]), numerical_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')), # 填充 Embarked, Title, CabinDeck, FareBin
            ('onehot', OneHotEncoder(handle_unknown='ignore'))]), categorical_features)
    ],
    remainder='passthrough' # 保留未处理的列（这里应该没有）
)


# 5. 模型选择 (示例使用 Random Forest)
# 您可以尝试其他模型，例如：
# model = LogisticRegression(solver='liblinear', random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5) # 示例参数，可以调优
# model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
# model = XGBClassifier(n_estimators=100, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss', random_state=42) # 如果安装了 xgboost
# model = LGBMClassifier(n_estimators=100, learning_rate=0.1, random_state=42) # 如果安装了 lightgbm


# 6. 构建完整的机器学习 Pipeline：预处理 -> 模型
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', model)])

# 7. 模型训练
print("\n开始训练模型...")
pipeline.fit(X_train, y_train)
print("模型训练完成！")


# ** 模型调优说明 (可选步骤，不包含在直接运行的代码中，但您可以在此基础上进行) **
# 例如，使用 GridSearchCV 寻找 RandomForestClassifier 的最佳参数
param_grid = {'classifier__n_estimators': [50, 100, 200],
              'classifier__max_depth': [3, 5, 10, None]} # 注意参数前缀是 pipeline step name '__' parameter name
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy') # cv=5 表示使用 5 折交叉验证
print("开始进行网格搜索调优...")
grid_search.fit(X_train, y_train)
print("最佳参数：", grid_search.best_params_)
print("交叉验证最佳得分：", grid_search.best_score_)
pipeline = grid_search.best_estimator_ # 使用找到的最佳模型



# 8. 模型预测
print("开始进行预测...")


predictions = pipeline.predict(X_test)
print("预测完成！")

# 9. 生成提交文件
# 创建一个 DataFrame，包含 PassengerId 和预测的 Survived 结果
submission_df = pd.DataFrame({'PassengerId': test_passenger_ids, 'Survived': predictions})

# 确保 Survived 列是整数类型 (0 或 1)
submission_df['Survived'] = submission_df['Survived'].astype(int)

# 将结果保存到 CSV 文件
submission_df.to_csv('submission.csv', index=False)

print("\n提交文件 'submission.csv' 已生成成功！")
print("文件预览:")
print(submission_df.head())

