from __future__ import annotations

import pytest

from bfmc_v2x.protocol import SemaphoreState
from bfmc_v2x.semaphore import (
    DEFAULT_PHASES,
    ApproachDecision,
    ObservedLight,
    Phase,
    SemaphoreCycle,
    SemaphoreTracker,
    StopLinePolicy,
)

PERIOD = sum(p.duration_s for p in DEFAULT_PHASES)


# ---- the light -----------------------------------------------------------


def test_cycle_walks_its_phases():
    cycle = SemaphoreCycle(light_id=1)

    assert cycle.state_at(0.0) == (SemaphoreState.GREEN, pytest.approx(7.0))
    assert cycle.state_at(6.5)[0] is SemaphoreState.GREEN
    assert cycle.state_at(7.5) == (SemaphoreState.YELLOW, pytest.approx(1.5))
    # Red runs from 9 s to 18 s, so 8 s remain at t = 10.
    assert cycle.state_at(10.0) == (SemaphoreState.RED, pytest.approx(8.0))


def test_cycle_repeats_every_period():
    cycle = SemaphoreCycle(light_id=1)

    for t in (0.0, 3.3, 8.1, 12.7, 17.9):
        now_state, now_ttl = cycle.state_at(t)
        later_state, later_ttl = cycle.state_at(t + PERIOD)
        assert now_state is later_state
        assert now_ttl == pytest.approx(later_ttl, abs=1e-6)


def test_cycle_is_a_pure_function_of_the_clock():
    early = SemaphoreCycle(light_id=1)
    rebooted = SemaphoreCycle(light_id=1)

    assert early.state_at(42.0) == rebooted.state_at(42.0)


def test_offset_axes_are_never_green_together():
    """The junction's safety property, checked exhaustively."""
    north_south = SemaphoreCycle(light_id=1, offset_s=0.0)
    east_west = SemaphoreCycle(light_id=3, offset_s=9.0)

    permissive = {SemaphoreState.GREEN, SemaphoreState.YELLOW}
    for step in range(int(PERIOD * 100)):
        t = step / 100.0
        ns, _ = north_south.state_at(t)
        ew, _ = east_west.state_at(t)
        assert not (ns in permissive and ew in permissive), "both permissive at t={:.2f}".format(t)


def test_every_approach_eventually_gets_a_green():
    east_west = SemaphoreCycle(light_id=3, offset_s=9.0)
    colours = {east_west.state_at(step / 10.0)[0] for step in range(int(PERIOD * 10))}

    assert SemaphoreState.GREEN in colours


def test_broadcast_carries_the_countdown():
    cycle = SemaphoreCycle(light_id=4)
    payload = cycle.broadcast_at(7.5)

    assert payload.light_id == 4
    assert payload.colour is SemaphoreState.YELLOW
    assert payload.ttl_s == pytest.approx(1.5, abs=0.001)


def test_zero_duration_phase_is_rejected():
    with pytest.raises(ValueError):
        Phase(SemaphoreState.RED, 0.0)


def test_empty_cycle_is_rejected():
    with pytest.raises(ValueError):
        SemaphoreCycle(light_id=1, phases=[])


# ---- the tracker ---------------------------------------------------------


def test_tracker_counts_the_countdown_down():
    tracker = SemaphoreTracker()
    tracker.ingest(SemaphoreCycle(1).broadcast_at(0.0), rx_time=100.0)

    observed = tracker.observe(1, now=100.5)

    assert observed.state is SemaphoreState.GREEN
    assert observed.ttl_s == pytest.approx(6.5, abs=0.01)


def test_tracker_discounts_time_spent_in_flight():
    tracker = SemaphoreTracker()
    payload = SemaphoreCycle(1).broadcast_at(5.0)

    tracker.ingest(payload, rx_time=100.0, link_age_s=0.2)
    observed = tracker.observe(1, now=100.0)

    assert observed.ttl_s == pytest.approx(1.8, abs=0.01)


def test_a_light_that_goes_quiet_goes_stale():
    tracker = SemaphoreTracker(stale_after_s=1.0)
    tracker.ingest(SemaphoreCycle(1).broadcast_at(0.0), rx_time=100.0)

    fresh = tracker.observe(1, now=100.4)
    stale = tracker.observe(1, now=102.0)

    assert fresh.ttl_s is not None and not fresh.is_stale
    # The colour survives; the countdown does not.
    assert stale.state is SemaphoreState.GREEN
    assert stale.ttl_s is None and stale.is_stale


def test_an_expired_countdown_withdraws_the_promise():
    tracker = SemaphoreTracker(stale_after_s=5.0)
    payload = SemaphoreCycle(1).broadcast_at(6.8)

    tracker.ingest(payload, rx_time=100.0)
    observed = tracker.observe(1, now=100.5)

    assert observed.ttl_s is None, "the light should have changed and we were not told"


def test_unknown_light_is_none():
    assert SemaphoreTracker().observe(99, now=0.0) is None


# ---- the policy ----------------------------------------------------------


@pytest.fixture()
def policy() -> StopLinePolicy:
    return StopLinePolicy(cruise_speed=0.35)


def light(state: SemaphoreState, ttl):
    return ObservedLight(light_id=1, state=state, ttl_s=ttl, age_s=0.05)


def test_green_with_time_to_spare_proceeds(policy):
    result = policy.evaluate(1.0, 0.35, light(SemaphoreState.GREEN, 6.0))

    assert result.decision is ApproachDecision.PROCEED
    assert result.target_speed == pytest.approx(0.35)


def test_green_that_expires_before_we_clear_plans_the_stop(policy):
    # 2 m away at 0.35 m/s is 5.7 s to the line plus 2.6 s to clear.
    result = policy.evaluate(2.0, 0.35, light(SemaphoreState.GREEN, 0.5))

    assert result.decision is not ApproachDecision.CLEAR
    assert "too late to clear" in result.reason


def test_green_without_a_countdown_is_approached_cautiously(policy):
    result = policy.evaluate(1.0, 0.35, light(SemaphoreState.GREEN, None))

    assert result.decision is ApproachDecision.PREPARE
    assert "countdown unknown" in result.reason


def test_red_that_clears_before_we_arrive_does_not_stop_us(policy):
    result = policy.evaluate(2.0, 0.35, light(SemaphoreState.RED, 0.3))

    assert result.decision is ApproachDecision.PROCEED
    assert result.target_speed == pytest.approx(0.35)


def test_red_stops_at_the_line(policy):
    result = policy.evaluate(0.08, 0.35, light(SemaphoreState.RED, 8.0))

    assert result.decision is ApproachDecision.STOP
    assert result.target_speed == 0.0


def test_red_sheds_speed_on_the_approach(policy):
    # Comfortable stopping distance from 0.35 m/s is only about 0.16 m.
    far = policy.evaluate(1.2, 0.35, light(SemaphoreState.RED, 8.0))
    near = policy.evaluate(0.20, 0.35, light(SemaphoreState.RED, 8.0))

    assert near.target_speed < far.target_speed
    assert near.decision is ApproachDecision.PREPARE


def test_red_about_to_change_creeps_rather_than_stopping(policy):
    # Arriving in 1.1 s at a light clearing in 1.4 s: creep, do not stop dead.
    result = policy.evaluate(0.4, 0.35, light(SemaphoreState.RED, 1.4))

    assert result.decision is ApproachDecision.PREPARE
    assert 0.0 < result.target_speed <= 0.35
    assert "creeping" in result.reason


def test_yellow_inside_the_dilemma_zone_commits(policy):
    result = policy.evaluate(0.10, 0.35, light(SemaphoreState.YELLOW, 1.0))

    assert result.decision is ApproachDecision.CLEAR
    assert result.target_speed == pytest.approx(0.35)


def test_yellow_with_room_to_stop_stops(policy):
    result = policy.evaluate(1.5, 0.35, light(SemaphoreState.YELLOW, 1.0))

    assert result.decision is not ApproachDecision.CLEAR
    assert "stoppable" in result.reason


def test_no_signal_is_treated_as_a_give_way(policy):
    result = policy.evaluate(0.08, 0.35, None)

    assert result.decision is ApproachDecision.STOP
    assert "give-way" in result.reason


def test_flashing_yellow_is_a_give_way(policy):
    result = policy.evaluate(0.08, 0.35, light(SemaphoreState.FLASHING_YELLOW, None))

    assert result.decision is ApproachDecision.STOP


def test_past_the_line_always_clears(policy):
    result = policy.evaluate(-0.2, 0.2, light(SemaphoreState.RED, 8.0))

    assert result.decision is ApproachDecision.CLEAR
    assert result.target_speed == pytest.approx(0.35)


def test_braking_distance_and_speed_target_are_consistent(policy):
    for distance in (0.1, 0.2, 0.35, 0.6, 1.0, 2.0):
        allowed = min(policy.cruise_speed, policy._speed_for(distance))
        needed = policy.braking_distance(allowed, policy.comfortable_decel)
        assert needed <= distance - policy.stop_margin_m + 1e-9, (
            "at {:.2f} m the policy allows {:.3f} m/s, which needs {:.3f} m".format(
                distance, allowed, needed
            )
        )


def test_the_approach_profile_is_monotonic(policy):
    red = light(SemaphoreState.RED, 8.0)
    speeds = [policy.evaluate(d / 100.0, 0.35, red).target_speed for d in range(8, 300)]

    assert all(a <= b + 1e-9 for a, b in zip(speeds, speeds[1:]))
