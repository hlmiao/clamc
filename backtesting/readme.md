# 金融行业某买方用户高性能计算解决方案

## Architectures

![](assets/arch.png)

* 完成L2行情数据清洗之后，使用GridSearch将在给定的参数值上训练给定的估计量。
* 遍历参数集寻找最优参数，找到给出最高（或最低，如果使用损失函数）得分的参数，选出最优的因子。
* 使用ECS Fargate为GridSearch提供算力，实现交叉验证。

## Results

![route](assets/results.png)

