import time
import torch
from sklearn.cluster import KMeans
import julei
# 读取数据
# print(a)
# # 聚类数量
start = time.time()
k = 20
model = KMeans(n_clusters=k)
# 训练模型
#########################################################################################
a = torch.randn(16, 12, 197, 64)
b = torch.randn(16, 12, 197, 64)
sum = 0
for i in range(a.size(0)):   # 将一个类别中的值来代替其他值
    for j in range(a.size(1)):
        data = a[i][j][:][:].tolist()
        start1 = time.time()
        # 分类中心点坐标
        # centers = model.cluster_centers_
        # 预测结果
        result = model.fit(data)
        result = result.labels_
        end0 = time.time()
        # print("模型时间：")
        sum = sum + (end0 - start1)
        # print(end0 - start1)
        # print(result)
        listResult = [[] for i in range(k)]  # 创建二维的列表，长度为k
        count = 0
        for m in result:  # 将分类的结果添加到listResult二维列表中
            listResult[m].append(count)
            count = count + 1
        # # print(listResult)
        for n in range(len(listResult)):
            for q in range(len(listResult[n])):
                a[i][j][listResult[n][q]][:] = a[i][j][listResult[n][0]][:]
c = torch.matmul(a, b.transpose(-1, -2))
c = torch.softmax(c, dim=-1)
end1 = time.time()
print('聚成%d类需要的时间为：'%k)
print(sum)
# print(a)
print("总时间：")
print(end1 - start)
###############################################################
# import numpy as np
# from sklearn.cluster import KMeans
# X = [[2, 2, 3],[5, 7, 7],[6, 2, 3],[5, 6, 7],[3, 2, 3],[6, 6, 7]]
# kmeans_model = KMeans(n_clusters=3).fit(X)
# labels = kmeans_model.labels_
# print(labels)