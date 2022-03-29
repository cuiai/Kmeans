import torch
from sklearn.cluster import KMeans
from collections import defaultdict
import time
# 读取数据
# print(a)
# # 聚类数量
start = time.time()
k = 20
model = KMeans(n_clusters=k)
# end0 = time.time()
# print("模型时间：")
# print(end0 - start)
a = torch.rand(16, 12, 197, 64)
c = torch.rand(16, 12, 197, 64)
b = torch.zeros(16, 12, k, 64)
d = torch.zeros(16, 12, 197, 197)
# print(a)
for i in range(a.size(0)):   # 将一个类别中的值来代替其他值
    for j in range(a.size(1)):
        data = a[i][j][:][:].tolist()
        model.fit(data)
        # 分类中心点坐标
        # centers = model.cluster_centers_
        # 预测结果
        result = model.predict(data)
        # print(result)
        listResult = [[] for i in range(k)]  # 创建二维的列表，长度为k
        count = 0
        for m in result:  # 将分类的结果添加到listResult二维列表中
            listResult[m].append(count)
            count = count + 1
        # print(listResult)
        for n in range(len(listResult)):
            b[i][j][n][:] = a[i][j][listResult[n][0]][:]
        # end1 = time.time()
        # print(i)
        # print(end1 - start)
        # print(b)
        attention_scores = torch.mm(b[i][j][:][:], c[i][j][:][:].transpose(-1, -2))
        attention_scores = torch.softmax(attention_scores, dim=-1)
        # print(attention_scores)
        for n in range(len(listResult)):
           for q in range(len(listResult[n])):
                d[i][j][listResult[n][q]][:] = attention_scores[n][:]
endtime = time.time()
print(endtime - start)
# print(d)




