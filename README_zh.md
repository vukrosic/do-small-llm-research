# 自回归Transformer消融研究

[English](README.md) | 中文版本

## 概述

本仓库包含了小规模自回归Transformer语言模型的综合消融研究，专注于资源受限环境下的快速实验和超参数优化。该研究在学习率、批量大小、序列长度和模型架构等多个维度上评估了六种不同的配置。

## 🚀 关键发现

- **批量大小缩放至关重要**：大批量配置实现了3.7倍的性能提升
- **学习率敏感性极强**：不当的学习率导致36倍的困惑度增加
- **存在效率权衡**：小模型提供44%的训练加速
- **短期训练揭示洞察**：每个配置仅需15分钟即可获得有意义的结果

## 📊 实验结果

| 配置 | 训练损失 | 准确率 | 困惑度 | 步数/秒 |
|------|----------|--------|--------|---------|
| 大批量 | **1.334** | **69.8%** | **4.93** | 63.12 |
| 高学习率 | 2.227 | 41.2% | 18.82 | 60.72 |
| 基线 | 3.642 | 30.4% | 48.85 | 64.51 |
| 短序列 | 3.875 | 34.9% | 43.69 | 64.57 |
| 小模型 | 4.053 | 29.4% | 77.25 | **87.73** |
| 低学习率 | 4.893 | 25.4% | 178.18 | 63.25 |

## 🏗️ 仓库结构

```
├── paper.tex                           # 研究论文 (LaTeX)
├── research/                           # 研究数据和结果
│   ├── research_data.json             # 完整实验数据
│   ├── performance_metrics.csv        # 性能指标
│   ├── research_report.md             # 详细分析报告
│   ├── training_loss_comparison.png   # 训练曲线
│   ├── final_performance_comparison.png
│   ├── efficiency_analysis.png
│   └── convergence_analysis.png
├── training_logs/                      # 原始训练日志
├── AR-Transformer-LLM.py             # 主训练脚本
├── enhanced_training.py               # 增强训练工具
├── research_data_collector.py         # 数据收集工具
├── plot_ablation_results.py          # 可视化脚本
└── colab_usage_guide.py              # Google Colab集成指南
```

## 🔧 快速开始

### 环境要求

```bash
pip install torch transformers datasets matplotlib seaborn pandas numpy
```

### 运行实验

1. **单配置训练**：
```bash
python AR-Transformer-LLM.py --config baseline
```

2. **完整消融研究**：
```bash
python enhanced_training.py --run-all-configs
```

3. **生成图表**：
```bash
python plot_ablation_results.py
```

### Google Colab

在Google Colab中轻松实验：
```python
# 在Colab中运行
exec(open('colab_usage_guide.py').read())
```

## 📈 测试配置

### 1. 基线配置
- **模型**：384维，6个头，12层
- **训练**：批量大小8，学习率3e-4，序列长度512
- **目的**：标准参考配置

### 2. 高学习率
- **修改**：学习率1e-3
- **结果**：快速收敛，存在一些不稳定性
- **洞察**：激进的学习率适用于短期训练

### 3. 低学习率
- **修改**：学习率1e-4
- **结果**：15分钟内收敛不佳
- **洞察**：保守的学习率不适合快速原型设计

### 4. 大批量
- **修改**：批量大小16，缩放学习率5e-4
- **结果**：最佳整体性能
- **洞察**：批量缩放对优化稳定性至关重要

### 5. 短序列
- **修改**：序列长度256，批量大小12
- **结果**：训练更快，性能降低
- **洞察**：上下文长度与效率的权衡

### 6. 小模型
- **修改**：256维，4个头，8层，批量大小12
- **结果**：最快训练（87.7步/秒）
- **洞察**：非常适合快速迭代

## 🎯 使用场景

### 研究与开发
- **快速原型设计**：15分钟内测试想法
- **超参数探索**：快速迭代周期
- **架构比较**：高效的模型选择

### 教育用途
- **学习Transformer**：动手实验
- **理解优化**：可视化训练动态
- **资源受限学习**：所有学生都可访问

### 生产应用
- **模型选择**：选择最优配置
- **资源规划**：了解计算需求
- **基线建立**：大型实验的起点

## 📊 可视化示例

仓库包含全面的可视化：

- **训练损失曲线**：比较收敛模式
- **性能指标**：最终准确率、困惑度比较
- **效率分析**：步数/秒、训练时间权衡
- **收敛分析**：早期与后期阶段动态

## 🔬 研究方法

### 数据集
- **来源**：HuggingFaceTB/smollm-corpus (cosmopedia-v2)
- **大小**：500个文档
- **分词器**：SmolLM-135M

### 训练设置
- **硬件**：NVIDIA Tesla T4 GPU
- **持续时间**：2000步（每个配置约15分钟）
- **评估**：每400步
- **精度**：混合精度（AMP）

### 跟踪指标
- 训练/验证损失和准确率
- 困惑度和置信度分数
- Top-5准确率
- 训练效率（步数/秒）
- 梯度范数和收敛模式

## 🚀 未来研究方向

### 即时扩展
1. **更长训练**：扩展到10K+步
2. **更多数据集**：在不同领域测试
3. **更大模型**：扩展到10亿+参数
4. **高级优化器**：Lion、AdamW变体

### 高级研究
1. **学习率调度**：预热策略、循环学习率
2. **正则化**：Dropout、权重衰减变化
3. **架构变体**：MoE、稀疏注意力
4. **多GPU扩展**：分布式训练模式

## 📝 引用

如果您在研究中使用此工作，请引用：

```bibtex
@article{rosic2025transformer_ablation,
  title={Ablation Study of Autoregressive Transformer Language Models: Hyperparameter Optimization and Architectural Trade-offs},
  author={Rosić, Vuk and Claude},
  journal={arXiv preprint},
  year={2025}
}
```

## 🤝 贡献

我们欢迎贡献！请查看我们的贡献指南：

1. Fork仓库
2. 创建功能分支
3. 添加您的实验或改进
4. 提交带有详细描述的拉取请求

### 贡献领域
- 新的配置实验
- 额外的可视化工具
- 性能优化
- 文档改进

## 📄 许可证

本项目采用MIT许可证 - 详见[LICENSE](LICENSE)文件。

## 🙏 致谢

- **HuggingFace**：提供SmolLM语料库和transformers库
- **PyTorch团队**：提供优秀的深度学习框架
- **Google Colab**：提供可访问的GPU资源
- **研究社区**：支持开放科学和可重现研究

## 📞 联系方式

- **作者**：Vuk Rosić
- **邮箱**：vukrosic1@gmail.com
- **机构**：Óbuda大学

---

**⭐ 如果您觉得这个仓库对您的研究有用，请给它点个星！**