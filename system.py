import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 用户-物品评分矩阵
ratings = np.array([
    [5, 4, 0, 0, 2],
    [3, 0, 0, 5, 0],
    [4, 0, 0, 3, 0],
    [0, 3, 4, 0, 0],
    [0, 0, 5, 4, 0]
])

# 计算用户之间的相似度
user_similarity = cosine_similarity(ratings)

# 预测用户对物品的评分
def predict_ratings(user_ratings, user_similarity):
    pred_ratings = user_similarity.dot(user_ratings) / np.array([np.abs(user_similarity).sum(axis=1)])
    return pred_ratings

predicted_ratings = predict_ratings(ratings, user_similarity)

# 输出预测结果
print("预测的用户-物品评分矩阵：")
print(predicted_ratings)
