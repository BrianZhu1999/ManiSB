# 案例一：平滑闭合耦合

这个案例用一个共享隐变量生成两个端点曲线。目标端点是三瓣闭合曲线，终止端点
是压缩并旋转后的闭合曲线。桥过程在两条曲线之间演化，模型学习桥坐标的几何校正。

评估使用 1,024 对端点和 60 次反向更新。包内的 `closed_coupling_data.npz`
包含端点、解析桥支撑、三头模型轨迹、校正后轨迹和逐样本指标所需数组。

运行导出检查：

```bash
python code/case1_smooth_closed/verify_exports.py
```

结果报告见 `results/case1_smooth_closed/REPORT_zh.md`，配置见
`configs/case1_swirl.json`。论文主图见
`figures/paper/fig_smooth_closed_coupling.pdf`。
