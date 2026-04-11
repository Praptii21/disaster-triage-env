"""
test_env.py
===========
Comprehensive tests for the Disaster Triage RL Environment.

Run with:
    uv run pytest test_env.py -v
or:
    .venv/Scripts/activate && pytest test_env.py -v
"""

import pytest
import numpy as np

from app.models import (
    Action,
    ActionType,
    Difficulty,
    EpisodeResult,
    Observation,
    ResourceBundle,
    ResourceType,
    Zone,
    DIFFICULTY_CONFIGS,
)
from app.env import DisasterTriageEnv, INFO_COST, ALLOCATION_COST, TEMPORAL_DECAY
from app.graders import DisasterTriageGrader


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def easy_env():
    return DisasterTriageEnv(difficulty=Difficulty.EASY, seed=42)

@pytest.fixture
def medium_env():
    return DisasterTriageEnv(difficulty=Difficulty.MEDIUM, seed=42)

@pytest.fixture
def hard_env():
    return DisasterTriageEnv(difficulty=Difficulty.HARD, seed=42)


# ===========================================================================
# 1. Reset Tests
# ===========================================================================

class TestReset:

    def test_reset_returns_observation(self, easy_env):
        obs = easy_env.reset()
        assert isinstance(obs, Observation)

    def test_reset_zone_count_easy(self, easy_env):
        obs = easy_env.reset()
        assert len(obs.zones) == 3

    def test_reset_zone_count_medium(self, medium_env):
        obs = medium_env.reset()
        assert len(obs.zones) == 5

    def test_reset_zone_count_hard(self, hard_env):
        obs = hard_env.reset()
        assert len(obs.zones) == 7

    def test_reset_step_count_zero(self, easy_env):
        obs = easy_env.reset()
        assert obs.step_count == 0

    def test_reset_resources_full(self, easy_env):
        obs = easy_env.reset()
        cfg = DIFFICULTY_CONFIGS[Difficulty.EASY]
        assert obs.available_resources.food     == cfg.initial_resources.food
        assert obs.available_resources.water    == cfg.initial_resources.water
        assert obs.available_resources.medicine == cfg.initial_resources.medicine

    def test_reset_done_false(self, easy_env):
        obs = easy_env.reset()
        assert obs.done is False

    def test_reset_clears_previous_episode(self, easy_env):
        easy_env.reset()
        easy_env.step(Action.request_info(0))
        easy_env.reset()
        assert easy_env.step_count == 0
        assert easy_env.accumulated_costs == 0.0
        assert easy_env.done is False

    def test_reset_seed_reproducible(self):
        env_a = DisasterTriageEnv(Difficulty.MEDIUM, seed=7)
        env_b = DisasterTriageEnv(Difficulty.MEDIUM, seed=7)
        obs_a = env_a.reset()
        obs_b = env_b.reset()
        for za, zb in zip(obs_a.zones, obs_b.zones):
            assert za.true_severity   == zb.true_severity
            assert za.urgency_signal  == zb.urgency_signal


# ===========================================================================
# 2. Observability Tests
# ===========================================================================

class TestObservability:

    def test_easy_all_zones_revealed(self, easy_env):
        obs = easy_env.reset()
        assert all(z.revealed for z in obs.zones), "EASY should start fully revealed"

    def test_medium_partial_zones_revealed(self, medium_env):
        obs = medium_env.reset()
        n_revealed = sum(z.revealed for z in obs.zones)
        assert 1 <= n_revealed <= len(obs.zones), "MEDIUM should be partially revealed"

    def test_hard_no_zones_revealed(self, hard_env):
        obs = hard_env.reset()
        assert not any(z.revealed for z in obs.zones), "HARD should start fully hidden"

    def test_unknown_zone_severity_masked(self, hard_env):
        obs = hard_env.reset()
        for zone in obs.zones:
            assert zone.known_severity == -1
            assert zone.known_demand   is None

    def test_data_completeness_easy(self, easy_env):
        obs = easy_env.reset()
        assert obs.data_completeness == 1.0

    def test_data_completeness_hard(self, hard_env):
        obs = hard_env.reset()
        assert obs.data_completeness == 0.0


# ===========================================================================
# 3. Action: request_info
# ===========================================================================

class TestRequestInfo:

    def test_request_info_reveals_zone(self, hard_env):
        hard_env.reset()
        obs, reward, done, info = hard_env.step(Action.request_info(0))
        zone = obs.zones[0]
        assert zone.revealed is True
        assert zone.known_severity in range(1, 6)
        assert zone.known_demand is not None

    def test_request_info_reward_is_negative(self, hard_env):
        hard_env.reset()
        _, reward, _, _ = hard_env.step(Action.request_info(0))
        expected = TEMPORAL_DECAY + INFO_COST  # -0.01 + -0.05 = -0.06
        assert abs(reward - expected) < 1e-6

    def test_request_info_step_increments(self, hard_env):
        hard_env.reset()
        hard_env.step(Action.request_info(0))
        assert hard_env.step_count == 1

    def test_request_info_duplicate_warns(self, hard_env):
        hard_env.reset()
        hard_env.step(Action.request_info(0))
        _, _, _, info = hard_env.step(Action.request_info(0))
        assert "warning" in info

    def test_request_info_invalid_zone_penalized(self, hard_env):
        hard_env.reset()
        _, reward, _, info = hard_env.step(Action.request_info(999))
        assert reward < TEMPORAL_DECAY + INFO_COST   # extra penalty applied
        assert "error" in info

    def test_request_info_updates_data_completeness(self, hard_env):
        hard_env.reset()
        obs, _, _, _ = hard_env.step(Action.request_info(0))
        expected = 1 / hard_env.config.num_zones
        assert abs(obs.data_completeness - expected) < 1e-6


# ===========================================================================
# 4. Action: allocate_resource
# ===========================================================================

class TestAllocateResource:

    def test_allocate_reduces_available_resources(self, easy_env):
        easy_env.reset()
        obs_before = easy_env._build_observation()
        easy_env.step(Action.allocate(0, ResourceType.FOOD, 20.0))
        assert easy_env.available_resources.food == obs_before.available_resources.food - 20.0

    def test_allocate_adds_to_zone(self, easy_env):
        easy_env.reset()
        easy_env.step(Action.allocate(0, ResourceType.WATER, 15.0))
        assert easy_env.zones[0].allocated.water == 15.0

    def test_allocate_reward_is_negative(self, easy_env):
        easy_env.reset()
        _, reward, _, _ = easy_env.step(Action.allocate(0, ResourceType.FOOD, 10.0))
        expected = TEMPORAL_DECAY + ALLOCATION_COST  # -0.01 + -0.10 = -0.11
        assert abs(reward - expected) < 1e-6

    def test_allocate_clamps_to_available(self, easy_env):
        easy_env.reset()
        easy_env.step(Action.allocate(0, ResourceType.FOOD, 90000.0))
        assert easy_env.available_resources.food == 0.0

    def test_allocate_invalid_zone(self, easy_env):
        easy_env.reset()
        _, reward, _, info = easy_env.step(Action.allocate(99, ResourceType.FOOD, 10.0))
        assert "error" in info

    def test_allocate_accumulates_multiple(self, easy_env):
        easy_env.reset()
        easy_env.step(Action.allocate(0, ResourceType.FOOD, 10.0))
        easy_env.step(Action.allocate(0, ResourceType.FOOD, 15.0))
        assert easy_env.zones[0].allocated.food == 25.0


# ===========================================================================
# 5. Action: finalize
# ===========================================================================

class TestFinalize:

    def test_finalize_sets_done(self, easy_env):
        easy_env.reset()
        _, _, done, _ = easy_env.step(Action.finalize())
        assert done is True and easy_env.done is True

    def test_finalize_reward_in_range(self, easy_env):
        easy_env.reset()
        _, reward, _, _ = easy_env.step(Action.finalize())
        assert 0.0 <= reward <= 1.0

    def test_finalize_returns_episode_result(self, easy_env):
        easy_env.reset()
        _, _, _, info = easy_env.step(Action.finalize())
        assert "episode_result" in info
        result = info["episode_result"]
        assert "total_reward" in result
        assert "final_score"  in result
        assert "steps_taken"  in result

    def test_finalize_score_breakdown(self, easy_env):
        easy_env.reset()
        _, _, _, info = easy_env.step(Action.finalize())
        bd = info["score_breakdown"]
        assert "prioritization" in bd
        assert "efficiency"     in bd
        assert "utilization"    in bd

    def test_finalize_after_done_raises(self, easy_env):
        easy_env.reset()
        easy_env.step(Action.finalize())
        with pytest.raises(RuntimeError):
            easy_env.step(Action.finalize())

    def test_finalize_better_with_allocation(self):
        """
        Smart allocation (match demand exactly on highest-severity zone)
        must beat immediate finalize without any allocation.
        Uses a seeded env so the highest-severity zone is deterministic.
        """
        SEED = 7

        # Baseline: finalize immediately (no allocations)
        env_a = DisasterTriageEnv(Difficulty.EASY, seed=SEED)
        env_a.reset()
        _, reward_no_alloc, _, _ = env_a.step(Action.finalize())

        # Smart agent: allocate exactly the true demand of the top zone
        env_b = DisasterTriageEnv(Difficulty.EASY, seed=SEED)
        env_b.reset()  # EASY = full observability → all zones revealed
        top = max(env_b.zones, key=lambda z: z.true_severity)
        env_b.step(Action.allocate(top.zone_id, ResourceType.FOOD,     top.true_demand.food))
        env_b.step(Action.allocate(top.zone_id, ResourceType.WATER,    top.true_demand.water))
        env_b.step(Action.allocate(top.zone_id, ResourceType.MEDICINE, top.true_demand.medicine))
        _, reward_alloc, _, _ = env_b.step(Action.finalize())

        assert reward_alloc >= reward_no_alloc, (
            f"Smart alloc ({reward_alloc:.4f}) should beat no-alloc ({reward_no_alloc:.4f})"
        )


# ===========================================================================
# 6. Step Budget
# ===========================================================================

class TestStepBudget:

    def test_auto_finalize_at_max_steps(self):
        """Env must auto-finalize when max_steps is reached."""
        env = DisasterTriageEnv(Difficulty.EASY, seed=0)
        env.reset()
        done = False
        for _ in range(env.config.max_steps):
            if done:
                break
            _, _, done, _ = env.step(Action.request_info(0))
        assert done is True

    def test_done_env_raises_on_step(self, easy_env):
        easy_env.reset()
        easy_env.step(Action.finalize())
        with pytest.raises(RuntimeError):
            easy_env.step(Action.request_info(0))


# ===========================================================================
# 7. Observation Tensor
# ===========================================================================

class TestObservationTensor:

    def test_tensor_shape_easy(self, easy_env):
        obs = easy_env.reset()
        tensor = obs.to_tensor()
        expected = easy_env.observation_shape  # (3*7 + 6,) = (27,)
        assert tensor.shape == expected

    def test_tensor_shape_hard(self, hard_env):
        obs = hard_env.reset()
        tensor = obs.to_tensor()
        expected = hard_env.observation_shape  # (7*7 + 6,) = (55,)
        assert tensor.shape == expected

    def test_tensor_dtype(self, easy_env):
        obs = easy_env.reset()
        assert obs.to_tensor().dtype == np.float32

    def test_to_dict_completeness(self, easy_env):
        obs = easy_env.reset()
        d = obs.to_dict()
        assert "zones"               in d
        assert "available_resources" in d
        assert "step_count"          in d
        assert "data_completeness"   in d
        assert "done"                in d


# ===========================================================================
# 8. Grader Tests
# ===========================================================================

class TestGrader:

    def setup_method(self):
        self.grader = DisasterTriageGrader()
        self.initial = ResourceBundle(food=100.0, water=100.0, medicine=100.0)
        self.remaining = ResourceBundle(food=0.0, water=0.0, medicine=0.0)

    def _make_zone(self, severity: int, demand: ResourceBundle, allocated: ResourceBundle) -> Zone:
        z = Zone(
            zone_id=0,
            true_severity=severity,
            true_demand=demand,
            urgency_signal=severity / 5.0,
        )
        z.allocated = allocated
        return z

    def test_perfect_allocation_high_score(self):
        demand = ResourceBundle(food=30.0, water=30.0, medicine=30.0)
        zones  = [self._make_zone(5, demand, ResourceBundle(food=30.0, water=30.0, medicine=30.0))]
        score, _ = self.grader.compute_final_score(zones, self.initial, self.remaining)
        assert score >= 0.7

    def test_zero_allocation_low_score(self):
        demand = ResourceBundle(food=30.0, water=30.0, medicine=30.0)
        zones  = [self._make_zone(5, demand, ResourceBundle())]
        score, _ = self.grader.compute_final_score(zones, self.initial, self.initial)
        assert score < 0.4

    def test_low_severity_over_allocation_penalized(self):
        demand = ResourceBundle(food=10.0, water=10.0, medicine=10.0)
        # Dumped 100 units into a low-severity zone
        zones  = [self._make_zone(1, demand, ResourceBundle(food=100.0, water=100.0, medicine=100.0))]
        score, bd = self.grader.compute_final_score(zones, self.initial, self.remaining)
        assert bd["prioritization"] < 0.5

    def test_score_always_in_range(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            sev    = int(rng.integers(1, 6))
            demand = ResourceBundle(food=float(rng.uniform(5, 50)),
                                    water=float(rng.uniform(5, 50)),
                                    medicine=float(rng.uniform(5, 50)))
            alloc  = ResourceBundle(food=float(rng.uniform(0, 100)),
                                    water=float(rng.uniform(0, 100)),
                                    medicine=float(rng.uniform(0, 100)))
            zones  = [self._make_zone(sev, demand, alloc)]
            score, _ = self.grader.compute_final_score(zones, self.initial, self.remaining)
            assert 0.0 <= score <= 1.0, f"Score out of range: {score}"

    def test_breakdown_keys(self):
        demand = ResourceBundle(food=20.0, water=20.0, medicine=20.0)
        zones  = [self._make_zone(3, demand, ResourceBundle(food=20.0, water=20.0, medicine=20.0))]
        _, bd = self.grader.compute_final_score(zones, self.initial, self.remaining)
        assert set(bd.keys()) == {"prioritization", "efficiency", "utilization", "final_score"}

    def test_explain_returns_zone_details(self):
        demand = ResourceBundle(food=20.0, water=20.0, medicine=20.0)
        zones  = [self._make_zone(3, demand, ResourceBundle(food=10.0, water=10.0, medicine=10.0))]
        report = self.grader.explain(zones, self.initial, self.remaining)
        assert "zone_details"    in report
        assert "score_breakdown" in report


# ===========================================================================
# 9. Full Episode Rollout
# ===========================================================================

class TestFullEpisode:

    def test_full_easy_episode(self):
        """Hand-crafted greedy rollout on EASY (full obs)."""
        env = DisasterTriageEnv(Difficulty.EASY, seed=0)
        obs = env.reset()

        # Sort zones by severity descending
        sorted_zones = sorted(obs.zones, key=lambda z: z.true_severity, reverse=True)

        # Distribute resources proportionally to severity
        total_severity = sum(z.true_severity for z in obs.zones)
        budget = obs.available_resources.food  # same for all resources

        for zone in sorted_zones:
            share = (zone.true_severity / total_severity) * budget
            env.step(Action.allocate(zone.zone_id, ResourceType.FOOD,     share / 3))
            env.step(Action.allocate(zone.zone_id, ResourceType.WATER,    share / 3))
            env.step(Action.allocate(zone.zone_id, ResourceType.MEDICINE, share / 3))

        _, reward, done, info = env.step(Action.finalize())
        assert done is True
        assert reward >= 0.0
        print(f"\n[EASY] Greedy rollout reward: {reward:.4f}")
        print(f"       Breakdown: {info['score_breakdown']}")

    def test_full_hard_episode_with_info_gathering(self):
        """HARD: reveal all zones then allocate to highest severity."""
        env = DisasterTriageEnv(Difficulty.HARD, seed=5)
        obs = env.reset()

        # Reveal all zones
        for zone in obs.zones:
            obs, _, done, _ = env.step(Action.request_info(zone.zone_id))
            if done:
                break

        # Sort by severity (we now know them)
        sorted_zones = sorted(env.zones, key=lambda z: z.true_severity, reverse=True)
        top_zone = sorted_zones[0]

        # Put everything into the most critical zone
        env.step(Action.allocate(top_zone.zone_id, ResourceType.FOOD,     33.0))
        env.step(Action.allocate(top_zone.zone_id, ResourceType.WATER,    33.0))
        env.step(Action.allocate(top_zone.zone_id, ResourceType.MEDICINE, 33.0))

        _, reward, done, info = env.step(Action.finalize())
        assert done is True
        assert reward >= 0.0
        print(f"\n[HARD] Info-then-allocate reward: {reward:.4f}")
        print(f"       Breakdown: {info['score_breakdown']}")

    def test_episode_result_dataclass_matches_info(self):
        env = DisasterTriageEnv(Difficulty.MEDIUM, seed=1)
        env.reset()
        env.step(Action.allocate(0, ResourceType.FOOD, 50.0))
        _, reward, done, info = env.step(Action.finalize())

        result: EpisodeResult = env.episode_result
        assert result is not None
        assert abs(result.total_reward - reward) < 1e-6
        assert result.total_reward == info["episode_result"]["total_reward"]


# ===========================================================================
# 10. Action Validation
# ===========================================================================

class TestActionValidation:

    def test_request_info_missing_zone_id(self):
        action = Action(action_type=ActionType.REQUEST_INFO)
        valid, msg = action.validate()
        assert not valid
        assert "zone_id" in msg

    def test_allocate_missing_zone_id(self):
        action = Action(action_type=ActionType.ALLOCATE_RESOURCE,
                        resource_type=ResourceType.FOOD, amount=10.0)
        valid, msg = action.validate()
        assert not valid
        assert "zone_id" in msg

    def test_allocate_zero_amount(self):
        action = Action(action_type=ActionType.ALLOCATE_RESOURCE,
                        zone_id=0, resource_type=ResourceType.FOOD, amount=0.0)
        valid, msg = action.validate()
        assert not valid
        assert "positive" in msg

    def test_allocate_negative_amount(self):
        action = Action(action_type=ActionType.ALLOCATE_RESOURCE,
                        zone_id=0, resource_type=ResourceType.WATER, amount=-5.0)
        valid, msg = action.validate()
        assert not valid

    def test_finalize_always_valid(self):
        action = Action.finalize()
        valid, _ = action.validate()
        assert valid

    def test_invalid_action_env_penalizes(self, hard_env):
        hard_env.reset()
        # Manually create an invalid action bypassing the factory
        bad = Action(action_type=ActionType.REQUEST_INFO, zone_id=None)
        _, reward, done, info = hard_env.step(bad)
        assert not info["valid"]
        assert reward < 0
        assert done is False


# ===========================================================================
# 11. Resource Bundle Math
# ===========================================================================

class TestResourceBundle:

    def test_addition(self):
        a = ResourceBundle(10, 20, 30)
        b = ResourceBundle(5, 10, 15)
        c = a + b
        assert c.food == 15 and c.water == 30 and c.medicine == 45

    def test_subtraction(self):
        a = ResourceBundle(10, 20, 30)
        b = ResourceBundle(5, 10, 15)
        c = a - b
        assert c.food == 5 and c.water == 10 and c.medicine == 15

    def test_can_afford_true(self):
        a = ResourceBundle(100, 100, 100)
        b = ResourceBundle(50, 50, 50)
        assert a.can_afford(b)

    def test_can_afford_false(self):
        a = ResourceBundle(10, 100, 100)
        b = ResourceBundle(50, 50, 50)
        assert not a.can_afford(b)

    def test_clamp_positive(self):
        a = ResourceBundle(-10, 5, -3)
        c = a.clamp_positive()
        assert c.food == 0.0 and c.water == 5.0 and c.medicine == 0.0

    def test_total(self):
        a = ResourceBundle(10, 20, 30)
        assert a.total() == 60.0

    def test_to_array_shape(self):
        a = ResourceBundle(1, 2, 3)
        arr = a.to_array()
        assert arr.shape == (3,) and arr.dtype == np.float32


# ===========================================================================
# 12. Factory & Make
# ===========================================================================

class TestFactory:

    def test_make_easy(self):
        env = DisasterTriageEnv.make("easy", seed=0)
        assert env.difficulty == Difficulty.EASY

    def test_make_hard(self):
        env = DisasterTriageEnv.make("hard", seed=0)
        assert env.difficulty == Difficulty.HARD

    def test_make_invalid_raises(self):
        with pytest.raises(ValueError):
            DisasterTriageEnv.make("ultra")
