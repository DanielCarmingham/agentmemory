const locks = new Map<string, Promise<void>>();

export function withKeyedLock<T>(
  key: string,
  fn: () => Promise<T>,
): Promise<T> {
  const prev = locks.get(key) ?? Promise.resolve();
  const next = prev.then(fn, fn);
  const cleanup = next.then(
    () => {},
    () => {},
  );
  locks.set(key, cleanup);
  cleanup.then(() => {
    if (locks.get(key) === cleanup) locks.delete(key);
  });
  return next;
}

// mem::summarize holds its own per-session lock across the provider call,
// which is far too coarse for a deletion to wait on. This narrower key
// covers only the summary-row write, so deletion paths serialize against
// that write without blocking behind an LLM round trip.
export function sessionWriteLockKey(sessionId: string): string {
  return `mem:session-write:${sessionId}`;
}
