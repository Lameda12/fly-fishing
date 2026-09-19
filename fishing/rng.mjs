// Seeded RNG for the scripted side of the task: bite times, spacing jitter, the
// policy's own action sampling, and which cached realization an episode splices
// in. Same algorithm as upstream's BrainEngine.random() (xorshift32) so a whole
// run is reproducible from one integer seed.

export function createRng(seed = 0x5eed1234) {
  let state = seed >>> 0;
  if (state === 0) state = 0x9e3779b9;
  const next = () => {
    state ^= state << 13;
    state >>>= 0;
    state ^= state >>> 17;
    state ^= state << 5;
    state >>>= 0;
    return state / 4294967296;
  };
  return {
    next,
    /** Integer in [0, bound). */
    int(bound) {
      return Math.floor(next() * bound) % Math.max(1, bound);
    },
    /** Uniform in [low, high). */
    range(low, high) {
      return low + next() * (high - low);
    },
    /**
     * Exponential deviate with the given mean. Bite spacing is drawn from this
     * so the gaps have no characteristic length a fixed-delay policy could lock
     * onto; the task then clamps it to a minimum spacing.
     */
    exponential(mean) {
      return -Math.log(1 - next()) * mean;
    },
    pick(items) {
      return items[Math.floor(next() * items.length) % items.length];
    },
    get state() {
      return state;
    },
  };
}

/**
 * Deterministic per-unit seeds derived from one run seed, so episode 40 gets the
 * same bite schedule whether or not episodes 1-39 ran.
 */
export function deriveSeed(runSeed, ...parts) {
  let hash = runSeed >>> 0;
  for (const part of parts) {
    const text = String(part);
    for (let i = 0; i < text.length; i++) {
      hash ^= text.charCodeAt(i);
      hash = Math.imul(hash, 0x01000193) >>> 0;
    }
    hash = (hash + 0x9e3779b9) >>> 0;
  }
  return (hash || 0x5eed1234) >>> 0;
}
