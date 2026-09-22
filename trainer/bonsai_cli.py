# bonsai_cli.py — the collaborative Bonsai + ewm-sm controller.
#
# The user talks to Bonsai through the lattice: every turn the controller
# proposes a context (full materialized memory, or the D-part novelty
# stream), Bonsai reasons over it, and the answer is ingested back into
# S(t). The context is a switchable, measurable artifact — see
# docs/BONSAI_COLLAB.md.
#
# Run:  python3 -m trainer.bonsai_cli
# Env:  BONSAI_URL (default http://127.0.0.1:8081)
#       EWM_BONSAI_WORK (default /home/alexmy/.cache/ewm-bonsai-cli)

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from .adapters import BonsaiAdapter, displacement_tokens, memory_tokens
from .ewm import EwmScene

DEFAULT_URL = "http://127.0.0.1:8081"
DEFAULT_WORK = "/home/alexmy/.cache/ewm-bonsai-cli"


@dataclass
class TurnRecord:
    turn: int
    query: str
    mode: str
    cap: int
    prefix_tokens: int
    prompt_tokens: int
    answer: str
    reasoning: str
    pop: int
    key: str


class BonsaiSession:
    """The collaborative controller: lattice state + context proposal + Bonsai."""

    def __init__(
        self,
        bonsai: BonsaiAdapter,
        ewm: EwmScene,
        work_dir: str,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ):
        self.bonsai = bonsai
        self.ewm = ewm
        self.work_dir = work_dir
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.union_path = os.path.join(work_dir, "session_union.jsonl")
        os.makedirs(work_dir, exist_ok=True)
        self.reset()

    def reset(self):
        with open(self.union_path, "w") as fh:
            fh.write("")
        self.seen: set = set()
        self.mem_tids: list[str] = []      # full materialized memory (restored order)
        self.comp_mem: list[str] = []      # D-part novelty stream
        self.turn = 0
        self.history: list[TurnRecord] = []
        self.policy = {"mode": "full", "cap": 24}

    # ── context proposal ────────────────────────────────────────────────
    def proposal_tids(self) -> list[str]:
        cap = int(self.policy["cap"])
        if self.policy["mode"] == "full":
            return self.mem_tids[-cap:]
        if self.policy["mode"] == "compact":
            return self.comp_mem[-cap:]
        raise ValueError(f"unknown context mode {self.policy['mode']!r}")

    def proposal_text(self) -> str:
        tids = self.proposal_tids()
        if not tids:
            return ""
        return self.bonsai.detokenize([int(t[3:]) for t in tids])

    # ── one turn of the collaboration loop ──────────────────────────────
    def ask(self, query: str) -> TurnRecord:
        prefix_tids = self.proposal_tids()
        if prefix_tids:
            prefix_text = self.bonsai.detokenize([int(t[3:]) for t in prefix_tids])
            prompt = f"Context memory ({self.policy['mode']}):\n{prefix_text}\n\nQuery: {query}"
        else:
            prompt = query

        full = self.bonsai.chat_full(prompt, max_tokens=self.max_tokens, temperature=self.temperature)
        answer = full["content"]
        reasoning = full["reasoning"]
        prompt_tokens = int(full["usage"].get("prompt_tokens", 0))

        tids = [f"tid{i}" for i in self.bonsai.tokenize(answer)]
        self.comp_mem.extend(displacement_tokens(self.seen, tids))
        with open(self.union_path, "a") as fh:
            fh.write(json.dumps({"id": self.turn + 1, "tokens": tids}) + "\n")

        mat = self.ewm.materialize(self.union_path)
        self.mem_tids = memory_tokens(mat["frames"][-1]["ordered"])
        uni = self.ewm.ingest(self.union_path)
        frame = uni["frames"][-1]

        rec = TurnRecord(
            turn=self.turn,
            query=query,
            mode=self.policy["mode"],
            cap=int(self.policy["cap"]),
            prefix_tokens=len(prefix_tids),
            prompt_tokens=prompt_tokens,
            answer=answer,
            reasoning=reasoning,
            pop=int(frame.get("pop", 0)),
            key=str(frame.get("key", "")),
        )
        self.history.append(rec)
        self.turn += 1
        return rec

    # ── state report ────────────────────────────────────────────────────
    def state_report(self) -> dict:
        uni = self.ewm.ingest(self.union_path)
        frames = uni.get("frames", [])
        last = frames[-1] if frames else {"pop": 0, "key": ""}
        report = {
            "turn": self.turn,
            "pop": int(last.get("pop", 0)),
            "key": str(last.get("key", ""))[:56],
            "mem_tids": len(self.mem_tids),
            "comp_tids": len(self.comp_mem),
            "policy": dict(self.policy),
            "noether": None,
        }
        if frames:
            noe = self.ewm.noether(self.union_path)
            report["noether"] = {
                "dp": noe.get("dp", [])[-1] if noe.get("dp") else None,
                "ind1": noe.get("ind1", [])[-1] if noe.get("ind1") else None,
                "ind2": noe.get("ind2", [])[-1] if noe.get("ind2") else None,
                "ind3": noe.get("ind3", [])[-1] if noe.get("ind3") else None,
            }
        return report


def _print_turn(rec: TurnRecord):
    ans = rec.answer.strip().replace("\n", " ")
    print(f"  [{rec.mode} cap={rec.cap}] prefix_tids={rec.prefix_tokens:3d} "
          f"prompt_tokens={rec.prompt_tokens:4d}  answer={ans[:80]!r}")


def main(argv: Optional[list[str]] = None) -> int:
    url = os.environ.get("BONSAI_URL", DEFAULT_URL)
    work = os.environ.get("EWM_BONSAI_WORK", DEFAULT_WORK)

    bonsai = BonsaiAdapter(base_url=url)
    ewm = EwmScene()
    session = BonsaiSession(bonsai, ewm, work)

    print("ewm-bonsai — Bonsai 2 27B through the ewm-sm lattice")
    print(f"  bonsai server: {url}  health={'ok' if bonsai.health() else 'DOWN'}")
    print("  context proposals: /context full|compact   /cap N   /policy   /state   /history   /reset   /quit")
    print()

    tty = sys.stdin.isatty()
    while True:
        if tty:
            try:
                line = input("ewm-bonsai> ").strip()
            except (EOFError, KeyboardInterrupt):
                break
        else:
            line = sys.stdin.readline()
            if not line:
                break
            line = line.strip()

        if not line:
            continue
        if line in ("/quit", "/exit"):
            break
        if line in ("/help",):
            print("  /context full|compact   /cap N   /policy   /state   /history   /reset   /quit")
            continue
        if line.startswith("/context "):
            mode = line.split()[1]
            if mode not in ("full", "compact"):
                print(f"  unknown mode {mode!r} (use full|compact)")
                continue
            session.policy["mode"] = mode
            print(f"  context proposal: {mode}, cap={session.policy['cap']}")
            continue
        if line.startswith("/cap "):
            try:
                cap = int(line.split()[1])
                assert cap > 0
            except Exception:
                print("  usage: /cap N")
                continue
            session.policy["cap"] = cap
            print(f"  context proposal: {session.policy['mode']}, cap={cap}")
            continue
        if line == "/policy":
            print(f"  policy: mode={session.policy['mode']} cap={session.policy['cap']}")
            continue
        if line == "/state":
            r = session.state_report()
            print(f"  turn={r['turn']}  S(t) pop={r['pop']}  key={r['key']}")
            print(f"  memory: full={r['mem_tids']} tids  compact(D-part)={r['comp_tids']} tids")
            if r["noether"]:
                noe = r["noether"]
                print(f"  noether: dp={noe['dp']} ind1={noe['ind1']} ind2={noe['ind2']} ind3={noe['ind3']}")
            print(f"  policy: {r['policy']}")
            continue
        if line == "/history":
            if not session.history:
                print("  (no turns yet)")
            for rec in session.history:
                print(f"  t={rec.turn} mode={rec.mode} cap={rec.cap} "
                      f"prefix_tids={rec.prefix_tokens} prompt_tokens={rec.prompt_tokens} "
                      f"pop={rec.pop}  ans={rec.answer[:50]!r}")
            continue
        if line == "/reset":
            session.reset()
            print("  session reset")
            continue

        rec = session.ask(line)
        _print_turn(rec)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
