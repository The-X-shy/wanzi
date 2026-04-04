# 数据目录说明

仓库已经包含主实验所需的两个文件：

- `metr-la.h5`
- `adj_mx.pkl`

这两份文件足够直接跑 `METR-LA` 主实验，不需要额外处理。

如果后续要做 `PEMS-BAY` 补充实验，请自行放入：

- `pems-bay.h5`
- `adj_mx_bay.pkl`

默认配置会在下面这些位置查找文件：

- `data/metr-la.h5`
- `data/adj_mx.pkl`
- `data/pems-bay.h5`
- `data/adj_mx_bay.pkl`
