"""Summarize decisions beyond an intact enemy tower in a policy action trace."""

import argparse
import json
from pathlib import Path


# Fallback for older traces without tower coordinates; verified in gamecore .stat.
ENEMY_TOWER_POSITION = {1: (11285.0, 11275.0), 2: (-11240.0, -11228.0)}


def summarize(run_dir):
    games = [json.loads(line) for line in (run_dir / "games.jsonl").open(encoding="utf-8")]
    rows = {index: {"game": index,
                    "mode": game["action_selection"],
                    "hero_id": game["hero_id"],
                    "opponent_id": game["opponent_id"],
                    "player_id": game["player_id"],
                    "outcome": game["outcome"],
                    "episode_decisions": game["episode_len"],
                    "deaths": game["final_snapshot"]["my"]["deaths"] -
                              game["initial_snapshot"]["my"]["deaths"],
                    "crossing_decisions": 0,
                    "move_decisions_beyond_tower": 0,
                    "deaths_beyond_tower": 0,
                    "first_cross_frame": None,
                    "last_cross_frame": None}
            for index, game in enumerate(games, 1)}
    source_counts = {"trace": 0, "stat_fallback": 0}

    with (run_dir / "action_trace.jsonl").open(encoding="utf-8") as trace:
        for line in trace:
            step = json.loads(line)
            before, after = step["before"], step["after"]
            player_id = before["player_id"]
            mine = next(hero for hero in before["heroes"] if hero["runtime_id"] == player_id)
            next_mine = next(hero for hero in after["heroes"] if hero["runtime_id"] == player_id)
            enemy = next(hero for hero in before["heroes"] if hero["runtime_id"] != player_id)
            camp = mine["camp"]
            if camp not in ENEMY_TOWER_POSITION or step["game"] not in rows:
                raise ValueError("未知阵营或对局编号")
            tower = next((organ for organ in before.get("towers", [])
                          if organ["camp"] == enemy["camp"]), None)
            if tower is not None and tower["x"] is not None and tower["z"] is not None:
                tower_x, tower_z, tower_hp = tower["x"], tower["z"], tower["hp"]
                source_counts["trace"] += 1
            else:
                tower_x, tower_z = ENEMY_TOWER_POSITION[camp]
                tower_hp = enemy["buildings"].get("21")
                source_counts["stat_fallback"] += 1
            forward = 1 if camp == 1 else -1
            crossed = (mine["hp"] > 0 and abs(mine["x"]) < 90000 and abs(mine["z"]) < 90000
                       and tower_hp is not None and tower_hp > 0
                       and forward * (mine["x"] - tower_x) > 0
                       and forward * (mine["z"] - tower_z) > 0)
            if not crossed:
                continue
            result = rows[step["game"]]
            result["crossing_decisions"] += 1
            if result["first_cross_frame"] is None:
                result["first_cross_frame"] = before["frame"]
            result["last_cross_frame"] = before["frame"]
            if step["action"][0] == 2:
                result["move_decisions_beyond_tower"] += 1
            if next_mine["hp"] <= 0:
                result["deaths_beyond_tower"] += 1

    return {"criterion": "alive hero beyond both enemy tower x/z coordinates while tower HP > 0",
            "tower_coordinate_sources": source_counts,
            "fallback_tower_coordinates_by_my_camp": ENEMY_TOWER_POSITION,
            "games": list(rows.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    run_dir = parser.parse_args().run_dir.resolve()
    result = summarize(run_dir)
    target = run_dir / "tower_crossing_summary.json"
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for game in result["games"]:
        print(game["mode"], game["opponent_id"], game["player_id"], game["outcome"],
              "crossing_decisions", game["crossing_decisions"],
              "deaths_beyond_tower", game["deaths_beyond_tower"])
    print(target)


if __name__ == "__main__":
    main()
