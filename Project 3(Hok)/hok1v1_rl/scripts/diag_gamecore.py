"""诊断 gamecore-server newGame 502 的真实原因。"""
import json
import sys

import requests

BASE = "http://127.0.0.1:23432/v2"


def post(ep, payload):
    try:
        r = requests.post(f"{BASE}/{ep}", json=payload, timeout=20)
        print(f"[{ep}] status={r.status_code} body={r.text[:800]}")
        return r
    except Exception as e:
        print(f"[{ep}] EXC {type(e).__name__}: {e}")
        return None


print("=== 简单端点 ===")
post("taskList", {})
post("exists", {"runtime_id": "hok1v1rl-train"})

print()
print("=== 构造 newGame ===")
from hok.hok1v1.hero_config import get_default_hero_config
from hok.common.camp import HERO_DICT, camp_iterator_1v1_roundrobin_camp_heroes
from hok.common.gamecore_client import GamecoreClient, SimulatorType

dhc = get_default_hero_config()
gc = GamecoreClient(
    server_addr="127.0.0.1:23432",
    gamecore_req_timeout=3000,
    default_hero_config=dhc,
    max_frame_num=20000,
    simulator_type=SimulatorType.RemoteRepeat,
)
camp = next(camp_iterator_1v1_roundrobin_camp_heroes(list(HERO_DICT.values())))
print("camp =", camp)

servers = [("127.0.0.1", 35150), None]

# 手工复刻 start_config，便于打印
start_config = {"hero_conf": [], "game_mode": camp["mode"], "max_frame_num": 20000}
for camp_id, hero_list in enumerate(camp["heroes"]):
    for hero_data in hero_list:
        hid = int(hero_data["hero_id"])
        info = dhc.get(hid, {})
        if not servers[camp_id] or not all(servers[camp_id]):
            req = {}
        elif camp_id == 0:
            req = {"ip": servers[camp_id][0], "port": servers[camp_id][1], "timeout": 3000}
        else:
            req = {"request_id": camp_id * len(hero_list)}
        start_config["hero_conf"].append({
            "hero_id": hid,
            "request_info": req,
            "skill_id": hero_data.get("skill_id", info.get("skill_id")),
            "symbol": hero_data.get("symbol", info.get("symbol")),
        })

for st in ("remote", "repeat", "remote_repeat"):
    payload = {"simulator_type": st, "runtime_id": "hok1v1rl-train",
               "simulator_config": start_config}
    print(f"--- simulator_type={st}")
    r = post("newGame", payload)
    if r is not None and r.status_code == 200:
        print("    >>> newGame 成功，随后清理")
        post("stopGame", {"runtime_id": "hok1v1rl-train"})
        break
