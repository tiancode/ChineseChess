# AlphaZero for Xiangqi (象棋)

A standalone PyTorch re-implementation of the AlphaGo Zero / AlphaZero
algorithm for Chinese Chess. No human games, no handcrafted evaluation: the
network learns purely from self-play guided by MCTS.

This subproject is independent of the Rust GUI. The only thing it borrows is
the **rules**, re-implemented in pure Python and validated to bit-for-bit
agree with the Rust engine's published perft values (44 / 1 920 / 79 666).

## The algorithm

```
        ┌─────────────────────────────────────────────┐
        │  self-play:  MCTS(网络) 对弈 → (s, π, z)      │
        │  train:      最小化  (z−v)² − πᵀlog p + c‖θ‖² │
        │  gate:       新网 vs 旧网, 胜率达标才晋级       │
        └──────────────────────┬──────────────────────┘
                       重复迭代 ┘
```

- **Network** (`net.py`) — ResNet trunk → policy head (8100 = 90×90 flat
  `from*to` actions) + value head (`tanh ∈ [-1,1]`).
- **MCTS** (`mcts.py`) — PUCT selection
  `argmax Q + c·P·√ΣN /(1+N)`, network leaf evaluation, Dirichlet noise at
  the root. Simulations descend a cloned `GameState` with `apply`/`undo`
  so repetition / no-capture counters stay correct in-search.
- **State encoding** (`encoding.py`) — canonicalised to the side to move
  (Black positions are rotated 180° and colour-swapped), 16 planes
  (14 piece + repetition + no-progress scalars).
- **Self-play / training / gate** — `selfplay.py`, `train.py`, `arena.py`,
  orchestrated by `pipeline.py`.

Terminal scoring follows the Rust engine exactly: a side with no legal reply
*loses* (mate or stalemate); threefold repetition and the 120-ply no-capture
rule are draws.

## Setup

```bash
pip install -r alphazero/requirements.txt   # CUDA torch + numpy
```

## Run

```bash
# 0. validate the rules port against the Rust perft numbers
python -m alphazero.tests.test_perft

# 1. fast end-to-end sanity check (tiny net, CPU, ~1 min)
python -m alphazero.pipeline --smoke

# 2. real training run (GPU). Override any Config field on the CLI:
python -m alphazero.pipeline
python -m alphazero.pipeline --iterations 200 --sims 400 --games_per_iter 80

# 3. play / watch a checkpoint
python -m alphazero.play --ckpt alphazero/checkpoints/best.pt --watch
python -m alphazero.play --ckpt alphazero/checkpoints/best.pt --human red
```

Checkpoints land in `alphazero/checkpoints/` (`best.pt` is the latest
gate-promoted network).

## Tuning (`config.py`)

| knob | meaning | bigger ⇒ |
|------|---------|----------|
| `channels`, `res_blocks` | network size | stronger, more VRAM/slower |
| `sims` | MCTS sims / move | stronger play, slower self-play |
| `games_per_iter` | self-play games / iteration | more/fresher data |
| `train_steps_per_iter` | optimisation steps / iteration | faster fitting, risk of overfit to stale data |
| `c_puct` | exploration constant | broader search |
| `dirichlet_alpha/eps` | root exploration noise | more opening variety |
| `arena_win_rate` | promotion threshold | stricter progress gate |
| `resign_disable_frac` | fraction of games played out (no resign) | better FP estimate, slower |
| `resign_target_fp` | tolerated resign false-positive rate | auto-tunes `resign_value` |

Optimisation uses **AdamW** (decoupled weight decay). Resignation has
**false-positive control**: `resign_disable_frac` of games disable
resignation and play to the end; if the would-resign side did not actually
lose it is counted a false positive, and (when `resign_auto_tune`) the
`resign_value` threshold is nudged each iteration to hold `resign_target_fp`.
The per-iteration `resign:` log line reports it.

`Config.smoke()` shrinks everything for the `--smoke` run.

## Expectations / honest caveats

This is a faithful, runnable implementation of the *algorithm*, not a
recipe for a superhuman engine. DeepMind's AlphaZero used thousands of TPUs;
on a single GPU with a pure-Python rules layer the bottleneck is self-play
throughput (Python move-gen + per-leaf NN calls), so a single machine will
produce a network that *clearly learns* and beats random/weak play within
hours, but reaching the strength of the Rust alpha-beta engine in `src/ai`
would take a long, sustained run. The knobs above let you trade strength for
wall-clock time.

Possible speedups (not implemented, to keep the code readable): leaf-batched
MCTS, multiprocess self-play workers, a compact (~2 000) move-label action
space, and a Cython/Rust rules binding.



  1. 启动正式训练(后台、分离、日志落盘)
  mkdir -p alphazero/checkpoints
  nohup setsid python3 -m alphazero.pipeline \
    --iterations 40 --games_per_iter 18 --sims 128 \
    --channels 128 --res_blocks 10 --max_game_len 200 \
    --train_steps_per_iter 300 --batch_size 256 \ 
    --arena_games 8 --arena_sims 64 --seed 0 \
    > alphazero/checkpoints/train.log 2>&1 &
    
    恢复训练命令(配置必须与原来一致,日志换个名避免覆盖历史):
  nohup setsid python3 -m alphazero.pipeline \
    --resume alphazero/checkpoints/best.pt \ 
    --iterations 40 --games_per_iter 18 --sims 128 \
    --channels 128 --res_blocks 10 --max_game_len 200 \
    --train_steps_per_iter 300 --batch_size 256 \ 
    --arena_games 8 --arena_sims 64 --seed 0 \
    >> alphazero/checkpoints/train_resume.log 2>&1 &


  2. 实时看进度
  tail -f alphazero/checkpoints/train.log
  
  3. 查进程 / 停止
  pgrep -af alphazero.pipeline      # 看是否在跑、PID
  pkill -f alphazero.pipeline       # 停止训练
  nvidia-smi                        # 看 GPU 占用
  
  预期(基于刚才实测)

  - 实测 ~42s/局(早期对局快速分胜负,非最坏情况)。
  - 粗估每迭代:自对弈 18 局 ≈ 13–18 min + 训练 300 步几分钟 + 轻量 arena 几分钟 ≈ ~20–30 min/迭代;40 迭代约 13–20 小时。这是单笔记本
  GPU(3050 Ti 4GB)+ 纯 Python 自对弈的现实开销。
  - 想更快看到信号:调小 --games_per_iter 或 --iterations;想更强:调大 --sims(最影响棋力,也最影响速度)。
  - checkpoint 落在 alphazero/checkpoints/best.pt 是 arena 闸门晋级的最新网络;随时可 python3 -m alphazero.play --ckpt 
  alphazero/checkpoints/best.pt --watch 观战。

  日志怎么读
  
  每迭代两行(及晋级行):
  [iter   3] R/B/D=8/7/3 samples=2310 buf=6840 | loss=2.10 (p=1.78 v=0.32) | sp=820s tr=140s
             resign: no-resign games=2 would-resign=1 fp=0 rate=0.00 thr=-0.92
             candidate PROMOTED (score 0.62) -> alphazero/checkpoints/best_iter003.pt
  健康信号:loss 总体下降、v(价值损失)从~0 升起再回落(开始学胜负而非全和)、R/B/D 出现非和棋且红黑大致均衡、fp rate 维持在 resign_target_fp
  附近。
  
  我不在你那个会话里看不到日志——跑起来后把几行 [iter …] 贴回来,我帮你判断是否健康、要不要调参。需要我把这条命令也写成一个 run_train.sh
  脚本吗?

