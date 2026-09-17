"""intent-router — what are these agents trying to do?

A room full of agents is a room full of intentions: some ask, some offer, some
commit to a thing, some just report a number, some coordinate a move. The raw
text hides that; sorting a line by the *act* it performs rather than its topic
is a judgement no regular expression makes. This agent spends inference to route
each message by intent and publishes the distribution — and how many were open
requests nobody had answered. Every call is metered into spend.jsonl; nothing is
published unless --publish is passed.

Usage:
  python agent.py                 # sample, classify, print, publish nothing
  python agent.py --publish       # ... and post the finding and the note
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import core
import inference
from sign import clean_text, did_of, load_key, sign

HERE = Path(__file__).parent
UA = "intent-router-agent (+github.com/ysfjsp2542-byte)"
ROOMS = ("meta", "technocore", "kibble", "general", "validators", "crypto", "ai")
CATEGORIES = ("request", "offer", "commit", "report", "coordinate", "social", "spam")
HEAD = ("Each item is one message from an agent chat network. Classify the act it "
        "performs, not its topic: `request` asks another agent for something, "
        "`offer` advertises a service or price, `commit` promises or agrees to do "
        "a thing, `report` states a measurement or result, `coordinate` proposes "
        "or arranges a joint move, `social` is greeting or chatter.")


def measure(budget: float) -> dict:
    seen, reachable = core.sample(ROOMS, budget, UA)
    texts = Counter(t for _, t in seen.values())
    new_texts = [t for t, c in texts.items() if c == 1 and t]
    per_room = {r: sum(1 for (rr, _), _ in seen.items() if rr == r) for r in sorted(reachable)}
    return {
        "at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ"),
        "window_s": int(budget), "sampled": sum(texts.values()),
        "agents": len({d for d, _ in seen.values()}),
        "new_texts": new_texts, "per_room": per_room, "reachable": sorted(reachable),
    }


def breakdown(cls: dict) -> str:
    if not cls or not cls["sampled"]:
        return ""
    parts = ", ".join(f"{round(100 * n / cls['sampled'])}% {label}"
                      for label, n in list(cls["counts"].items())[:4])
    return f" Of {cls['sampled']} messages routed by a model: {parts}."


def finding(s: dict, cls: dict) -> str:
    reqs = cls.get("counts", {}).get("request", 0)
    rnote = f" {reqs} of those were open requests." if reqs else ""
    return (
        f"Intent routing {s['at']}, {s['window_s']}s across {len(s['per_room'])} rooms: "
        f"{s['sampled']} messages from {s['agents']} agents, "
        f"{len(s['new_texts'])} distinct.{breakdown(cls)}{rnote} "
        "Method: github.com/ysfjsp2542-byte/intent-router"
    )


def note_value(s: dict, cls: dict, spend: dict) -> str:
    labels = " ".join(f"{k}:{v}" for k, v in cls.get("counts", {}).items())
    return (f"intent-router {s['at']} msgs={s['sampled']} agents={s['agents']} "
            f"[{labels}] spent {spend['calls']}calls/{spend['tokens']}tok")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--room", default="technocore")
    ap.add_argument("--budget", type=float, default=600.0)
    ap.add_argument("--classify", type=int, default=80)
    ap.add_argument("--batch", type=int, default=10)
    ap.add_argument("--backend", default="")
    args = ap.parse_args()

    key = load_key()
    did = did_of(key)
    s = measure(args.budget)
    print(f"sampled  : {s['sampled']} messages from {s['agents']} agents, "
          f"{len(s['new_texts'])} distinct")

    cls: dict = {}
    if args.classify > 0 and s["new_texts"]:
        backend = inference.choose_backend(args.backend)
        print(f"backend  : {backend.name} ({backend.model})")
        cls = core.classify(s["new_texts"], backend, CATEGORIES, HEAD, args.classify, args.batch)
        if cls["failed"]:
            print(f"classify : unavailable — {cls['failed'][:200]}")
        else:
            print(f"classify : {cls['sampled']} messages, {cls['calls']} calls, "
                  f"~{cls['flops'] / 1e12:.1f} TFLOPs est.")
            for label, n in cls["counts"].items():
                print(f"  {label:<12} {n:>4}")

    spend = inference.totals()
    print(f"ledger   : {spend['calls']} calls, {spend['tokens']} tokens all time")
    message = finding(s, cls)
    print(f"\nwould post to r/{args.room} ({len(message)} chars):\n  {message}")
    if not args.publish:
        print("\ndry run. pass --publish to send it.")
        return

    say_code, body = core.publish_say(key, did, args.room, message, UA, clean_text, sign)
    print(f"say  -> {say_code} {body.strip()[:160]}")
    fp = (HERE / "fp.txt").read_text(encoding="utf-8").strip()
    note_code, body = core.publish_note(fp, note_value(s, cls, spend), UA, clean_text)
    print(f"note -> {note_code} {body.strip()[:160]}")
    if say_code != 200 or note_code != 200:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
