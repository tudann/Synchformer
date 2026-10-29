# Synchformer 复现（对照 D16）

> 目的：在固定回归短 clip 上跑 Synchformer 偏移预测，旁路对照现有 SyncNet + 清单，判断能否优化 D16。  
> 不需要裁判模型；推理约需数 GB 显存。不要和 Qwen 裁判抢同一张卡。

上游仓库已克隆到本目录父级：`/keyan/research-eval-group/lt_self/Synchformer`（[v-iashin/Synchformer](https://github.com/v-iashin/Synchformer)，MIT）。

## 目录

| 路径 | 用途 |
|------|------|
| `setup_env.sh` | 创建本地 `.venv` 并安装推理依赖（本机无 conda，改用 pip） |
| `download_weights.sh` | 预下载 AudioSet / LRS3 权重到 `../logs/sync_models/` |
| `run_d16_cases.py` | 批量跑固定回归 D16 成片，写出 JSONL |
| `outputs/` | 批量结果落盘 |

## 推荐权重

| exp_name | 训练数据 | 用途 |
|----------|----------|------|
| `24-01-04T16-39-21` | AudioSet | 动作音 / 环境音等一般事件同步（先跑这个） |
| `23-12-23T18-33-57` | LRS3 | 口型向对照 |

首次跑 `example.py` 或本目录脚本时，缺权重会自动从官方 a3s.fi 下载；也可先执行 `download_weights.sh`。

## 一步步

```bash
cd /keyan/research-eval-group/lt_self/Synchformer/reproduce
bash setup_env.sh
source .venv/bin/activate
bash download_weights.sh          # 可选，先下 AudioSet 权重
# 单条冒烟（选空闲卡，例如 cuda:0 若有空闲）
cd ..
python example.py \
  --exp_name 24-01-04T16-39-21 \
  --vid_path ../eval_dem_13_20/data/固定回归case/16_audio_visual_sync/t2va/160101/video-group-new.mp4 \
  --offset_sec 0.0 \
  --device cuda:0

# 批量 D16（默认 video-group-new.mp4 + AudioSet 权重）
cd reproduce
python run_d16_cases.py --device cuda:0
```

`--offset_sec 0`：不人为错位，看模型预测的偏移是否接近 0（越接近越同步）。结果写在 `outputs/`。

## 和框架的关系

- 输入：成片 mp4（含音轨），不要题库、不要源视频、不要 VLM。
- 输出：预测偏移（秒）+ top-k 概率；与 SyncNet / 清单并排，不进对外权重。
- AVGen-Bench 的 Sync 模块就是同一套 AudioSet 权重，不必再装微软整仓。
