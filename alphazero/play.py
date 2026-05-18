"""Play against, or watch, a trained network in the terminal.

  python -m alphazero.play --ckpt alphazero/checkpoints/best.pt --watch
  python -m alphazero.play --ckpt alphazero/checkpoints/best.pt --human red

Human input is four integers ``from_file from_rank to_file to_rank`` with
file 0..8 and rank 0..9 (rank 0 is the top, Black's home -- the same
coordinate system the rules use).
"""

from __future__ import annotations

import argparse

from .xiangqi.board import (
    RED, BLACK, FILES, RANKS, idx, file_of, rank_of, decode_piece,
)
from .xiangqi.game import GameState
from .config import Config
from .net import load_net
from .mcts import Evaluator, run_mcts, select_move

_GLYPH = {
    (0, RED): "帥", (0, BLACK): "將", (1, RED): "仕", (1, BLACK): "士",
    (2, RED): "相", (2, BLACK): "象", (3, RED): "馬", (3, BLACK): "馬",
    (4, RED): "車", (4, BLACK): "車", (5, RED): "炮", (5, BLACK): "砲",
    (6, RED): "兵", (6, BLACK): "卒",
}


def render(game: GameState):
    print("\n   " + " ".join(f"{f}" for f in range(FILES)))
    for r in range(RANKS):
        row = []
        for f in range(FILES):
            code = int(game.board[idx(f, r)])
            if code == 0:
                row.append("·")
            else:
                kind, color = decode_piece(code)
                row.append(_GLYPH[(kind, color)])
        marker = "  <- 楚河漢界" if r == 5 else ""
        print(f"{r:>2} " + " ".join(row) + marker)
    side = "红" if game.side_to_move == RED else "黑"
    print(f"轮到: {side}   {'(将军!)' if game.in_check() else ''}")


def ai_move(game, ev, cfg):
    _root, visits = run_mcts(game, ev, cfg, add_noise=False)
    return select_move(visits, 0.0)


def human_move(game):
    legal = set(game.legal_moves())
    while True:
        try:
            raw = input("你的走法 (ff fr tf tr，q 退出): ").strip()
            if raw in ("q", "quit"):
                raise SystemExit
            ff, fr, tf, tr = (int(x) for x in raw.split())
            mv = (idx(ff, fr), idx(tf, tr))
        except SystemExit:
            raise
        except Exception:
            print("  输入无效，例如:  1 9 2 7")
            continue
        if mv in legal:
            return mv
        print("  非法走法，请重试。")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--watch", action="store_true", help="网络自对弈观战")
    p.add_argument("--human", choices=["red", "black"],
                   help="你执红或执黑与网络对弈")
    p.add_argument("--sims", type=int, default=200)
    args = p.parse_args()

    cfg = Config()
    cfg.sims = args.sims
    net = load_net(args.ckpt, cfg.device)
    ev = Evaluator(net, cfg)

    game = GameState()
    human = None
    if args.human == "red":
        human = RED
    elif args.human == "black":
        human = BLACK

    render(game)
    ply = 0
    while True:
        tv = game.terminal_value()
        if tv is not None:
            if tv == 0.0:
                print("\n和棋。")
            else:
                w = "黑" if game.side_to_move == RED else "红"
                print(f"\n{w}方胜。")
            break
        if ply >= cfg.max_game_len:
            print("\n步数上限，判和。")
            break

        if human is not None and game.side_to_move == human:
            mv = human_move(game)
        else:
            ev.clear()
            mv = ai_move(game, ev, cfg)
            print(f"\n网络走子: {file_of(mv[0])},{rank_of(mv[0])} -> "
                  f"{file_of(mv[1])},{rank_of(mv[1])}")
        game.apply(mv)
        render(game)
        ply += 1


if __name__ == "__main__":
    main()
