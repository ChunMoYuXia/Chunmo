"""连续对局 → 整局或固定长度片段 → GAE → LSTM chunks。"""
from collections import deque
import numpy as np
import time
from datetime import datetime
import torch
from .action_mask import LABEL_SIZE_LIST, effective_legal_action
from .reward import REWARD_COMPONENT_NAMES, compute_value_targets, terminal_reward_terms
from .reward_diagnostics import REWARD_DETECTION_KEYS, step_reward_detection
from .policy_diagnostics import PolicyDiagnostics, merge_policy_diagnostics
from .behavior_diagnostics import BEHAVIOR_KEYS, position_frame, step_evidence


def near_fountain(state, camp, radius=6000):
    """仅诊断：存活英雄距己方泉水建筑中心的平面距离；缺失坐标返回 None。"""
    req = state.get("req_pb")
    if req is None:
        return None
    hero = next((h for h in req.hero_list if int(h.camp) == camp), None)
    fountain = next((o for o in req.organ_list if int(o.camp) == camp and int(o.type) == 23), None)
    if hero is None or fountain is None or hero.hp <= 0:
        return None
    position, center = getattr(hero, "location", None), getattr(fountain, "location", None)
    if any(p is None or not hasattr(p, "x") or not hasattr(p, "z") for p in (position, center)):
        return None
    return (position.x-center.x)**2 + (position.z-center.z)**2 <= radius**2

BATCH_KEYS = ("features", "legal_actions", "actions", "log_probs", "values",
              "rewards", "dones", "advantages", "returns", "head_weights", "valid_mask")


def action_head_weights(state, button):
    """button 决定哪些动作头实际生效；缺失掩码应报错而非悄悄训练错误动作。"""
    masks = state.get("sub_action_mask")
    if isinstance(masks, dict):
        row = masks.get(button, masks.get(str(button)))
    elif masks is not None:
        row = masks[button]
    else:
        row = None
    if row is None or np.asarray(row).shape != (6,):
        raise ValueError("缺少有效的 sub_action_mask[button]，请检查 SDK 版本")
    weights = np.asarray(row, dtype=np.float32)
    if not np.isin(weights, [0, 1]).all() or weights[0] != 1:
        raise ValueError("动作头掩码应为 0/1，button 头必须有效")
    return weights


def pad_chunks(sequence, length):
    """末尾填零只为统一张量形状；valid_mask 会排除这些填充位置。"""
    array = np.asarray(sequence, dtype=np.float32)
    count = len(array)
    padding = (-count) % length
    array = np.pad(array, [(0, padding)] + [(0, 0)] * (array.ndim - 1))
    return array.reshape(-1, length, *array.shape[1:])


class EpisodeCollector:
    """保留一局的状态、记忆和统计；collect 返回后，对局仍可继续。

    一次 collect 最多采样 limit 步，并且不跨越终局。每个片段独立计算
    GAE，未终局时 bootstrap；完整对局统计在真正终局后才交给训练入口。
    """
    def __init__(self, runner, matchup=None, evaluation=False, burn_in=0):
        self.runner = runner
        self.started = time.monotonic()
        self.state = runner.reset(matchup=matchup, evaluation=evaluation)
        self.start_timings = {"reset_seconds": time.monotonic() - self.started,
                              **runner.reset_timings}
        self.hidden = runner.network.init_hidden(1, runner.device)
        self.initial_snapshot = runner._last_snapshot
        self.previous_snapshot = self.initial_snapshot
        self.history = deque(maxlen=burn_in)
        self.samples = []
        self.decisions = 0
        self.finished = False
        self.versions = []
        runner.network.eval()
        self._progress()

    @property
    def active(self):
        return not self.finished

    def _progress(self):
        if self.runner.progress_callback:
            match = self.runner.current_matchup.to_dict() if self.runner.current_matchup else {}
            self.runner.progress_callback(episode_decisions=self.decisions, **match,
                **self.runner.episode_info, control=self.runner.control)

    @torch.no_grad()
    def refresh_memory(self):
        """新权重重放最近的观测；更早的记忆仍是近似，不清零正在进行的对局。"""
        if not self.active or not self.history:
            return 0.0
        started = time.monotonic()
        hidden = self.history[0][1]
        history = []
        for feature, _ in self.history:
            history.append((feature, hidden))
            tensor = torch.as_tensor(feature, device=self.runner.device).reshape(1, 1, -1)
            hidden = self.runner.inference(tensor, hidden)["lstm_state_new"]
        self.history.clear()
        self.history.extend(history)
        self.hidden = hidden
        return time.monotonic() - started

    @torch.no_grad()
    def collect(self, limit, version=None, deterministic=False, random_policy=False,
                finalize_limit=False):
        if not self.active or limit <= 0:
            raise ValueError("只能从未结束的对局采样正数步")
        runner, cfg = self.runner, self.runner.cfg
        started, first_decision = time.monotonic(), self.decisions
        timings = {"reset_seconds": 0.0, "close_game_seconds": 0.0,
                   "start_game_seconds": 0.0, "inference_seconds": 0.0, "env_step_seconds": 0.0}
        if self.decisions == 0:
            timings.update(self.start_timings)
        if version is not None and (not self.versions or self.versions[-1] != version):
            self.versions.append(version)
        diagnostic = PolicyDiagnostics()
        state_diagnostics = {key: 0 for key in (
            "income_counter_decreases", "wallet_decreases", "experience_decreases",
            "missing_building_frames", "position_known_decisions", "fountain_decisions",
            "hero_skill_attempts", "hero_skill_attempts_with_known_position",
            "hero_skill_attempts_in_fountain", "attack_attempts", "attack_without_enemy_target",
            "attack_target_constraint_decisions", "attack_button_blocked_decisions")}
        state_diagnostics.update(dict.fromkeys(BEHAVIOR_KEYS, 0))
        data = {key: [] for key in BATCH_KEYS if key not in ("advantages", "returns", "valid_mask")}
        initial_states = []
        initial_snapshot = runner._last_snapshot
        components = {key: 0.0 for key in (*REWARD_COMPONENT_NAMES, "time", "terminal", "win_speed")}
        reward_detection = dict.fromkeys(REWARD_DETECTION_KEYS, 0)
        counts = [np.zeros(size, dtype=np.int64) for size in LABEL_SIZE_LIST]
        outcome, evidence, terminated = None, None, False
        for index in range(limit):
            state = self.state
            feature = runner.build_observation(state)
            raw_legal = np.asarray(state["legal_action"], dtype=np.float32).reshape(-1)
            legal = effective_legal_action(state, cfg.action)
            runner.last_effective_legal = legal
            raw_attack = raw_legal[76:].reshape(12, 8)[3]
            state_diagnostics["attack_target_constraint_decisions"] += int(
                not np.array_equal(legal[76:].reshape(12, 8)[3], raw_attack))
            state_diagnostics["attack_button_blocked_decisions"] += int(raw_legal[3] == 1 and legal[3] == 0)
            if index % runner.T == 0:
                initial_states.append(tuple(s.detach().cpu().clone() for s in self.hidden))
            # LSTM 返回新张量，不会就地改写旧记忆；保留引用即可，无须逐步复制。
            self.history.append((feature, self.hidden))
            inference_started = time.monotonic()
            action, logp, self.hidden, value_prediction = runner.sample_action(
                feature, legal, self.hidden, deterministic, random_policy)
            if int(action[0]) == 3:
                state_diagnostics["attack_attempts"] += 1
                state_diagnostics["attack_without_enemy_target"] += int(
                    not raw_attack[[1, 3, 4, 5, 6, 7]].any())
            timings["inference_seconds"] += time.monotonic() - inference_started
            diagnostic.add(legal, runner.last_button_probabilities, int(action[0]))
            in_fountain = near_fountain(state, runner._camps[0])
            if in_fountain is not None:
                state_diagnostics["position_known_decisions"] += 1
                state_diagnostics["fountain_decisions"] += int(in_fountain)
            if int(action[0]) in (4, 5, 6, 10):
                state_diagnostics["hero_skill_attempts"] += 1
                if in_fountain is not None:
                    state_diagnostics["hero_skill_attempts_with_known_position"] += 1
                    state_diagnostics["hero_skill_attempts_in_fountain"] += int(in_fountain)
            weights = action_head_weights(state, int(action[0]))
            compressed = np.concatenate((legal[:76], legal[76:].reshape(12, 8)[action[0]]))
            before_frame = position_frame(state)
            before_snapshot = runner._last_snapshot
            step_started = time.monotonic()
            self.state, reward, terminated, step_components = runner.step(action)
            timings["env_step_seconds"] += time.monotonic() - step_started
            current_snapshot = runner._last_snapshot
            for key, contribution in step_reward_detection(before_snapshot, current_snapshot, int(action[0])).items():
                reward_detection[key] += contribution
            observed = step_evidence(before_frame, position_frame(self.state), before_snapshot,
                                     current_snapshot, int(action[0]))
            for key, contribution in observed.items():
                state_diagnostics[key] += contribution
            if self.previous_snapshot is not None and current_snapshot is not None:
                for field, name in (("money", "income_counter_decreases"),
                                    ("wallet", "wallet_decreases"), ("exp", "experience_decreases")):
                    before = self.previous_snapshot["my"].get(field)
                    after = current_snapshot["my"].get(field)
                    if before is not None and after is not None and after < before:
                        state_diagnostics[name] += 1
            if current_snapshot is not None:
                if any(v is None for side in ("my_organs", "en_organs")
                       for v in current_snapshot[side].values()):
                    state_diagnostics["missing_building_frames"] += 1
                self.previous_snapshot = current_snapshot
            for key, contribution in step_components.items():
                components[key] += contribution
            if terminated:
                outcome = runner._terminal_outcome(self.state)
                evidence = runner._outcome_evidence(self.state, outcome)
                terminal = terminal_reward_terms(outcome, getattr(self.state.get("req_pb"), "frame_no", None),
                    runner.episode_info.get("initial_frame"), cfg.env.max_frame_num, cfg.reward)
                reward += sum(terminal.values())
                for key, contribution in terminal.items():
                    components[key] += contribution
            values = dict(features=feature, legal_actions=compressed, actions=action,
                          log_probs=logp, values=value_prediction, rewards=reward,
                          dones=float(terminated), head_weights=weights)
            for key, item in values.items():
                data[key].append(item)
            for head, choice in enumerate(action):
                counts[head][choice] += 1
            self.decisions += 1
            if runner.progress_callback:
                runner.progress_callback(episode_decisions=self.decisions)
            if terminated:
                break

        last_value = 0.0
        if not terminated:
            feature = torch.as_tensor(runner.build_observation(self.state), device=runner.device)
            last_value = float(runner.inference(feature.reshape(1, 1, -1), self.hidden)["value"].item())
        data["advantages"], data["returns"] = compute_value_targets(
            data["rewards"], cfg.ppo.gamma, data["values"], last_value,
            cfg.ppo.lamda, data["dones"])
        count = len(data["features"])
        data["valid_mask"] = np.ones(count, dtype=np.float32)
        result = {key: pad_chunks(values, runner.T) for key, values in data.items()}
        result["actions"] = result["actions"].astype(np.int64)
        result["lstm_state_init"] = tuple(torch.cat([s[i] for s in initial_states], dim=1) for i in (0, 1))
        reward = float(sum(data["rewards"]))
        if not np.isclose(reward, sum(components.values()), rtol=1e-5, atol=1e-4):
            raise RuntimeError("奖励日志分量与轨迹总回报不一致")
        ended = terminated or finalize_limit
        reason = ("crystal_destroyed" if outcome in ("win", "lose") else "draw" if outcome == "draw"
                  else "unknown_gameover" if terminated else "collector_limit" if finalize_limit
                  else "fragment_boundary")
        match = runner.current_matchup.to_dict() if runner.current_matchup else {}
        stats = {**match, **runner.episode_info, "control": runner.control,
                 "action_rule": cfg.action.rule,
                 "ended_at": datetime.now().astimezone().isoformat(timespec="milliseconds"),
                 "outcome": outcome or ("unknown" if ended else "ongoing"), "win": outcome == "win",
                 "episode_len": count, "truncated": finalize_limit and not terminated,
                 "termination_reason": reason, "bootstrap_value": last_value, "reward": reward,
                 "initial_snapshot": initial_snapshot, "final_snapshot": runner._last_snapshot,
                 "state_diagnostics": state_diagnostics, "reward_components": components,
                 "reward_detection": reward_detection,
                 "outcome_evidence": evidence, "chunk_padded_steps": (-count) % runner.T,
                 "last_step_valid": True, "terminal_in_batch": terminated,
                 "episode_decisions_start": first_decision, "episode_decisions_end": self.decisions,
                 "policy_version": version, "policy_diagnostics": diagnostic.to_dict(),
                 "timings": {**timings, "trajectory_seconds": time.monotonic() - started},
                 "action_hist": {f"head_{i}": c.tolist() for i, c in enumerate(counts)}}
        self.samples.append(stats)
        self.finished = ended
        result.update(sample_stats=stats, episode_stats=self.summary() if ended else None,
                      episode_len=count, episode_finished=ended, win=outcome == "win")
        return result

    def summary(self, interrupted=False):
        """合并整局统计；片段边界不增加胜负、终局奖励或对局数。"""
        if not self.samples:
            return None
        result = dict(self.samples[-1])
        result.update(episode_len=self.decisions, initial_snapshot=self.initial_snapshot,
                      episode_decisions_start=0,
                      reward=sum(s["reward"] for s in self.samples),
                      chunk_padded_steps=sum(s["chunk_padded_steps"] for s in self.samples),
                      fragment_count=len(self.samples), policy_versions=self.versions.copy(),
                      policy_version_start=self.versions[0] if self.versions else None,
                      policy_version_end=self.versions[-1] if self.versions else None)
        for key in ("reward_components", "state_diagnostics", "reward_detection", "timings"):
            result[key] = {name: sum(s[key].get(name, 0) for s in self.samples)
                           for name in self.samples[-1][key]}
        result["timings"]["trajectory_seconds"] = time.monotonic() - self.started
        result["action_hist"] = {name: np.sum([s["action_hist"][name] for s in self.samples], axis=0).tolist()
                                 for name in self.samples[-1]["action_hist"]}
        result["policy_diagnostics"] = merge_policy_diagnostics(self.samples)
        if interrupted:
            result.update(outcome="interrupted", win=False, truncated=True,
                          termination_reason="training_interrupted", terminal_in_batch=False,
                          ended_at=datetime.now().astimezone().isoformat(timespec="milliseconds"))
        return result


@torch.no_grad()
def collect_episode(runner, max_decisions=None, deterministic=False,
                    random_policy=False, matchup=None, evaluation=False):
    """兼容整局采样和诊断用的外部截断；二者也使用相同的单步逻辑。"""
    cfg = runner.cfg
    limit = max_decisions or cfg.env.max_decisions or cfg.env.max_frame_num // cfg.env.predict_frequency + 8
    if limit <= 0:
        raise ValueError("max_decisions 必须为正")
    collector = EpisodeCollector(runner, matchup, evaluation)
    return collector.collect(limit, deterministic=deterministic, random_policy=random_policy,
                             finalize_limit=True)


def collect_batch(runner, min_decisions, on_episode=None):
    """按真实决策数凑 batch，不能把 padding 算成采样进度。"""
    episodes, count = [], 0
    while count < min_decisions:
        episode = runner.collect_trajectory()
        episodes.append(episode)
        count += episode["episode_len"]
        if on_episode:
            on_episode(episode["episode_stats"])
    return join_trajectories(episodes), [e["episode_stats"] for e in episodes]


def join_trajectories(episodes):
    """按完整序列拼接；训练与离线诊断共用相同的 batch 格式。"""
    batch = {key: torch.as_tensor(np.concatenate([e[key] for e in episodes])) for key in BATCH_KEYS}
    batch["lstm_state_init"] = tuple(torch.cat([e["lstm_state_init"][i] for e in episodes], 1) for i in (0, 1))
    return batch
