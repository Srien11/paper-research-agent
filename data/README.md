# 本地数据目录

此目录只保留数据契约和说明，不提交论文 PDF 或生成索引。

预期本地结构：

```text
data/
  raw/          # 本地论文或外部冻结语料的受控映射
  processed/    # 页面、章节、元素和 chunk
  indexes/      # 向量与 BM25 索引
  evaluations/
    datasets/   # 可公开或脱敏的评测题
    runs/       # 每次实验结果，不提交大文件
```

任何准备提交的数据必须先确认 `storage_class`、许可证和第三方材料状态。
