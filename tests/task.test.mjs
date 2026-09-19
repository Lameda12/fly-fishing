import assert from "node:assert/strict";

import {
  TASK,
  WINDOWS_PER_EPISODE,
  buildSchedule,
  createEpisode,
  loomAt,
  stimulusAt,
} from "../fishing/task.mjs";
import { STIMULUS_CHANNELS } from "../fishing/brain-host.mjs";

// --- the schedule -----------------------------------------------------------
{
  const schedule = buildSchedule(12345);
  assert.ok(schedule.length >= 4, "an episode should get several bites");
  assert.ok(
    schedule.every((event) => event.type === "bite"),
    "commit 1 has one event type",
  );
  assert.ok(schedule[0].atMs >= TASK.leadInMs, "no bite inside the lead-in");
  assert.ok(
    schedule.at(-1).atMs < TASK.episodeMs,
    "a bite must not start after the episode ends",
  );

  for (let i = 1; i < schedule.length; i++) {
    const gap = schedule[i].atMs - schedule[i - 1].atMs;
    assert.ok(gap >= TASK.minBiteGapMs, `gap ${gap} ms is under the minimum`);
  }
}

// The minimum gap is what lets an episode be spliced from cached responses, so
// it has to exceed the recorded response length, not merely be nonzero.
{
  const responseMs = 2_800; // cache.RESPONSE_MS
  assert.ok(
    TASK.minBiteGapMs >= responseMs,
    "minimum bite gap must cover a whole recorded response",
  );
}

// --- schedules differ between episodes and repeat within one ----------------
{
  const a = buildSchedule(1);
  const b = buildSchedule(2);
  assert.deepEqual(buildSchedule(1), a, "a seed reproduces its schedule");
  assert.notDeepEqual(a, b, "different seeds must give different schedules");

  // A fixed-delay policy would win if every episode had the same gaps.
  const gaps = (s) => s.slice(1).map((e, i) => e.atMs - s[i].atMs);
  const spread = new Set([...gaps(a), ...gaps(b)].map((g) => Math.round(g / 100)));
  assert.ok(spread.size > 4, "bite spacing must actually vary");
}

// --- stimulus ---------------------------------------------------------------
{
  const events = [{ atMs: 1000, type: "bite" }];
  assert.equal(loomAt(events, 999), 0);
  assert.equal(loomAt(events, 1000), TASK.biteLoomHz);
  assert.equal(loomAt(events, 1000 + TASK.biteDurationMs - 1), TASK.biteLoomHz);
  assert.equal(loomAt(events, 1000 + TASK.biteDurationMs), 0);

  const stimulus = stimulusAt(events, 1000);
  for (const key of Object.keys(stimulus)) {
    assert.ok(STIMULUS_CHANNELS.includes(key), `${key} is not a real stimulus channel`);
  }
  assert.equal(stimulus.hungerHz, TASK.backgroundHungerHz);
}

// --- scoring ----------------------------------------------------------------
// Hooking inside the window lands the fish.
{
  const episode = createEpisode([{ atMs: 1000, type: "bite" }]);
  assert.equal(episode.step(950, 1).outcome, "snap", "early is a snapped line");
  assert.ok(!episode.canDecide(1000), "a hook starts a re-cast");
}
{
  const episode = createEpisode([{ atMs: 1000, type: "bite" }]);
  const result = episode.step(1200, 1);
  assert.equal(result.outcome, "catch");
  assert.equal(result.reward, TASK.rewardCatch);
  assert.equal(episode.finish().caught, 1);
}
// The edge of the hook window: the last millisecond counts, the next does not.
{
  const late = createEpisode([{ atMs: 1000, type: "bite" }]);
  assert.equal(late.step(1000 + TASK.hookWindowMs - 1, 1).outcome, "catch");
  const tooLate = createEpisode([{ atMs: 1000, type: "bite" }]);
  assert.equal(tooLate.step(1000 + TASK.hookWindowMs, 1).outcome, "snap");
}
// Waiting through a bite is a miss and costs nothing.
{
  const episode = createEpisode([{ atMs: 1000, type: "bite" }]);
  for (let t = 0; t < 3000; t += TASK.windowMs) episode.step(t, 0);
  const summary = episode.finish();
  assert.equal(summary.caught, 0);
  assert.equal(summary.missed, 1);
  assert.equal(summary.snapped, 0);
  assert.equal(summary.totalReward, 0);
}
// One fish cannot be caught twice.
{
  const episode = createEpisode([{ atMs: 1000, type: "bite" }]);
  assert.equal(episode.step(1100, 1).outcome, "catch");
  // Past the re-cast, the same bite's window is long gone anyway; force the
  // claim check by using a bite that is still nominally live.
  const second = createEpisode([{ atMs: 1000, type: "bite" }]);
  second.step(1000, 1);
  assert.ok(!second.canDecide(1100), "still re-casting");
  assert.equal(second.step(1100, 1).outcome, "recast", "no decision during a re-cast");
  assert.equal(second.finish().hooks, 1, "a re-cast window is not a hook");
}
// The re-cast blocks decisions for exactly as long as it says.
{
  const episode = createEpisode([]);
  episode.step(0, 1);
  const free = TASK.windowMs + TASK.recastMs;
  assert.ok(!episode.canDecide(free - 1), "still blocked one millisecond early");
  assert.ok(episode.canDecide(free), "free again at the end of the re-cast");
}

// --- summary arithmetic -----------------------------------------------------
{
  const events = [
    { atMs: 1000, type: "bite" },
    { atMs: 5000, type: "bite" },
    { atMs: 9000, type: "bite" },
  ];
  const episode = createEpisode(events);
  episode.step(1100, 1); // catch
  episode.step(5100, 1); // catch
  episode.step(20000, 1); // snap
  const summary = episode.finish();
  assert.equal(summary.bites, 3);
  assert.equal(summary.caught, 2);
  assert.equal(summary.snapped, 1);
  assert.equal(summary.missed, 1);
  assert.equal(summary.hooks, 3);
  assert.equal(summary.totalReward, 2 * TASK.rewardCatch + TASK.rewardSnap);
  assert.equal(summary.catchRate, 2 / 3);
  assert.equal(summary.hookPrecision, 2 / 3);
  assert.equal(summary.falseHooksPerMinute, 1, "60 s episode, one snapped line");
}

// --- window grid ------------------------------------------------------------
assert.equal(WINDOWS_PER_EPISODE, 1200);
assert.equal(TASK.hookWindowMs % TASK.windowMs, 0, "the hook window must be a whole number of windows");

console.log("task: all assertions passed");
