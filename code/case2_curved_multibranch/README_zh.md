# 案例二：带曲率的多分支耦合

这个案例用两个分支和曲率参数构造端点耦合。模型从终止端点条件出发恢复初始端点，
并比较单头、三头和三头加学习式桥坐标校正。

已归档结果使用 `kappa=0.6`、20 次反向更新、4,096 对端点和 `rho_max=0.02`。
训练和评估的完整数值见 `configs/` 与 `provenance/`。

检查数据包：

```bash
python code/case2_curved_multibranch/verify_case2_package.py --root .
```

案例二结果报告见 `results/case2_curved_multibranch/metrics_summary.json`，论文图
见 `figures/paper/fig3_multibranch_linear_v5.pdf`。

项目包记录了一个训练种子，以及单头和三头各自的参数量、训练步数与批大小。
