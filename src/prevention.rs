//! Circuit breakers and rate limiters for storm prevention.
//!
//! # Biological Analogs
//!
//! | Rust type | Biological analog | Mechanism |
//! |-----------|-------------------|-----------|
//! | [`CircuitBreaker`] | Tocilizumab (IL-6 receptor blocker) | Blocks the amplification signal once failure rate exceeds threshold |
//! | [`RateLimiter`] | JAK inhibitor (e.g. Baricitinib) | Occupies the ATP-binding pocket — consumes signal capacity |

use nexcore_homeostasis_primitives::enums::CircuitState;
use serde_json::Value;
use std::collections::VecDeque;
use std::time::Duration;
use tokio::sync::Mutex;
use tokio::time::Instant;
use tracing::{error, info, warn};

// =============================================================================
// CircuitBreaker internals
// =============================================================================

#[derive(Debug, Clone, Copy)]
struct StateChange {
    from: CircuitState,
    to: CircuitState,
}

struct CircuitBreakerInner {
    state: CircuitState,
    failure_count: u32,
    success_count: u32,
    last_failure_time: Option<Instant>,
    half_open_requests: u32,
    total_requests: u64,
    blocked_requests: u64,
    state_changes: VecDeque<StateChange>,
}

impl CircuitBreakerInner {
    fn new() -> Self {
        Self {
            state: CircuitState::Closed,
            failure_count: 0,
            success_count: 0,
            last_failure_time: None,
            half_open_requests: 0,
            total_requests: 0,
            blocked_requests: 0,
            state_changes: VecDeque::with_capacity(20),
        }
    }

    fn transition_to(&mut self, new_state: CircuitState, name: &str) {
        let old = self.state;
        self.state = new_state;
        match new_state {
            CircuitState::HalfOpen => {
                self.half_open_requests = 0;
                self.success_count = 0;
            }
            CircuitState::Closed => {
                self.failure_count = 0;
            }
            CircuitState::Open => {}
        }
        if self.state_changes.len() >= 20 {
            self.state_changes.pop_front();
        }
        self.state_changes.push_back(StateChange {
            from: old,
            to: new_state,
        });
        warn!(circuit = name, from = ?old, to = ?new_state, "circuit state transition");
    }
}

// =============================================================================
// CircuitBreaker
// =============================================================================

/// Three-state circuit breaker that blocks amplification cascades.
///
/// State machine:
/// - `Closed` — normal; trips to `Open` after `failure_threshold` consecutive failures.
/// - `Open` — blocks all requests; auto-transitions to `HalfOpen` after `recovery_timeout`.
/// - `HalfOpen` — allows limited test requests; closes on `success_threshold` successes,
///   reopens on any failure.
///
/// # Example
///
/// ```rust,no_run
/// # tokio_test::block_on(async {
/// use nexcore_homeostasis_storm::prevention::CircuitBreaker;
/// use std::time::Duration;
///
/// let cb = CircuitBreaker::new("database", 5, 2, Duration::from_secs(30), 3);
/// if cb.allow_request().await {
///     // do work …
///     cb.record_success().await;
/// }
/// # });
/// ```
pub struct CircuitBreaker {
    /// Human-readable name used in log messages.
    pub name: String,
    /// Consecutive failures before opening (default 5).
    pub failure_threshold: u32,
    /// Consecutive successes in `HalfOpen` required to close (default 2).
    pub success_threshold: u32,
    /// Duration spent `Open` before testing recovery (default 30 s).
    pub recovery_timeout: Duration,
    /// Maximum test requests allowed when `HalfOpen` (default 3).
    pub half_open_max_requests: u32,
    inner: Mutex<CircuitBreakerInner>,
}

impl CircuitBreaker {
    /// Create with explicit configuration.
    pub fn new(
        name: impl Into<String>,
        failure_threshold: u32,
        success_threshold: u32,
        recovery_timeout: Duration,
        half_open_max_requests: u32,
    ) -> Self {
        Self {
            name: name.into(),
            failure_threshold,
            success_threshold,
            recovery_timeout,
            half_open_max_requests,
            inner: Mutex::new(CircuitBreakerInner::new()),
        }
    }

    /// Create with sensible defaults (5 failures / 2 successes / 30 s timeout / 3 half-open).
    pub fn with_defaults(name: impl Into<String>) -> Self {
        Self::new(name, 5, 2, Duration::from_secs(30), 3)
    }

    /// Whether a request should be allowed through.
    ///
    /// - `Closed` → always `true`.
    /// - `Open` → `false`; auto-transitions to `HalfOpen` if `recovery_timeout` elapsed.
    /// - `HalfOpen` → `true` up to `half_open_max_requests`, then `false`.
    pub async fn allow_request(&self) -> bool {
        let mut g = self.inner.lock().await;
        g.total_requests += 1;

        // Auto-transition Open → HalfOpen once timeout elapses.
        if g.state == CircuitState::Open {
            if let Some(t) = g.last_failure_time {
                if t.elapsed() >= self.recovery_timeout {
                    g.transition_to(CircuitState::HalfOpen, &self.name);
                }
            }
        }

        match g.state {
            CircuitState::Closed => true,
            CircuitState::Open => {
                g.blocked_requests += 1;
                false
            }
            CircuitState::HalfOpen => {
                if g.half_open_requests < self.half_open_max_requests {
                    g.half_open_requests += 1;
                    true
                } else {
                    g.blocked_requests += 1;
                    false
                }
            }
        }
    }

    /// Record a successful operation.
    ///
    /// In `HalfOpen`, accumulates towards `success_threshold` before closing.
    /// In `Closed`, resets the failure counter.
    pub async fn record_success(&self) {
        let mut g = self.inner.lock().await;
        match g.state {
            CircuitState::HalfOpen => {
                g.success_count += 1;
                if g.success_count >= self.success_threshold {
                    g.transition_to(CircuitState::Closed, &self.name);
                }
            }
            CircuitState::Closed => {
                g.failure_count = 0;
            }
            CircuitState::Open => {}
        }
    }

    /// Record a failed operation.
    ///
    /// In `Closed`, increments the failure counter and trips to `Open` if threshold reached.
    /// In `HalfOpen`, immediately returns to `Open`.
    pub async fn record_failure(&self) {
        let mut g = self.inner.lock().await;
        g.last_failure_time = Some(Instant::now());
        match g.state {
            CircuitState::HalfOpen => {
                g.transition_to(CircuitState::Open, &self.name);
            }
            CircuitState::Closed => {
                g.failure_count += 1;
                if g.failure_count >= self.failure_threshold {
                    g.transition_to(CircuitState::Open, &self.name);
                }
            }
            CircuitState::Open => {}
        }
    }

    /// Force the circuit open immediately (emergency intervention).
    pub async fn force_open(&self) {
        let mut g = self.inner.lock().await;
        g.last_failure_time = Some(Instant::now());
        g.transition_to(CircuitState::Open, &self.name);
        error!(circuit = %self.name, "circuit FORCE OPENED");
    }

    /// Force the circuit closed (manual recovery reset).
    pub async fn force_close(&self) {
        let mut g = self.inner.lock().await;
        g.transition_to(CircuitState::Closed, &self.name);
        warn!(circuit = %self.name, "circuit force closed");
    }

    /// Current circuit state.
    pub async fn state(&self) -> CircuitState {
        self.inner.lock().await.state
    }

    /// Statistics snapshot as a JSON value.
    pub async fn get_statistics(&self) -> Value {
        let g = self.inner.lock().await;
        let block_rate = g.blocked_requests as f64 / g.total_requests.max(1) as f64;
        serde_json::json!({
            "name": self.name,
            "state": g.state,
            "failure_count": g.failure_count,
            "total_requests": g.total_requests,
            "blocked_requests": g.blocked_requests,
            "block_rate": block_rate,
            "transition_count": g.state_changes.len(),
        })
    }
}

// =============================================================================
// RateLimiter internals
// =============================================================================

struct RateLimiterInner {
    tokens: f64,
    last_refill: Instant,
    tokens_per_second: f64,
    bucket_capacity: f64,
    total_requests: u64,
    rejected_requests: u64,
}

impl RateLimiterInner {
    fn new(tokens_per_second: f64, bucket_capacity: f64) -> Self {
        Self {
            tokens: bucket_capacity,
            last_refill: Instant::now(),
            tokens_per_second,
            bucket_capacity,
            total_requests: 0,
            rejected_requests: 0,
        }
    }

    fn refill(&mut self) {
        let elapsed = self.last_refill.elapsed().as_secs_f64();
        let added = elapsed * self.tokens_per_second;
        self.tokens = (self.tokens + added).min(self.bucket_capacity);
        self.last_refill = Instant::now();
    }
}

// =============================================================================
// RateLimiter
// =============================================================================

/// Token-bucket rate limiter — throttles signal propagation.
///
/// Tokens refill at `tokens_per_second`. Each [`acquire`](RateLimiter::acquire)
/// consumes one token. When the bucket is empty, requests are rejected.
///
/// # Example
///
/// ```rust,no_run
/// # tokio_test::block_on(async {
/// use nexcore_homeostasis_storm::prevention::RateLimiter;
///
/// let rl = RateLimiter::new("api_endpoint", 100.0, 500.0);
/// if rl.acquire().await {
///     // process request …
/// }
/// # });
/// ```
pub struct RateLimiter {
    /// Human-readable name used in log messages.
    pub name: String,
    inner: Mutex<RateLimiterInner>,
}

impl RateLimiter {
    /// Create a new rate limiter.
    ///
    /// * `tokens_per_second` — refill rate.
    /// * `bucket_capacity` — maximum token count.
    pub fn new(name: impl Into<String>, tokens_per_second: f64, bucket_capacity: f64) -> Self {
        Self {
            name: name.into(),
            inner: Mutex::new(RateLimiterInner::new(tokens_per_second, bucket_capacity)),
        }
    }

    /// Try to consume one token. Returns `true` if the request is allowed.
    pub async fn acquire(&self) -> bool {
        let mut g = self.inner.lock().await;
        g.total_requests += 1;
        g.refill();
        if g.tokens >= 1.0 {
            g.tokens -= 1.0;
            true
        } else {
            g.rejected_requests += 1;
            false
        }
    }

    /// Dynamically adjust the token refill rate.
    pub async fn set_rate(&self, new_rate: f64) {
        let mut g = self.inner.lock().await;
        let old = g.tokens_per_second;
        g.tokens_per_second = new_rate;
        info!(limiter = %self.name, old_rate = old, new_rate, "rate adjusted");
    }

    /// Current number of tokens in the bucket (after refilling elapsed time).
    pub async fn current_tokens(&self) -> f64 {
        let mut g = self.inner.lock().await;
        g.refill();
        g.tokens
    }

    /// Bucket fill ratio: `1.0` = full, `0.0` = empty.
    pub async fn utilization(&self) -> f64 {
        let mut g = self.inner.lock().await;
        g.refill();
        let cap = g.bucket_capacity;
        g.tokens / cap
    }

    /// Current refill rate in tokens per second.
    pub async fn tokens_per_second(&self) -> f64 {
        self.inner.lock().await.tokens_per_second
    }

    /// Bucket capacity (maximum token count).
    pub async fn bucket_capacity(&self) -> f64 {
        self.inner.lock().await.bucket_capacity
    }

    /// Statistics snapshot as a JSON value.
    pub async fn get_statistics(&self) -> Value {
        let mut g = self.inner.lock().await;
        g.refill();
        let rejection_rate = g.rejected_requests as f64 / g.total_requests.max(1) as f64;
        serde_json::json!({
            "name": self.name,
            "tokens_per_second": g.tokens_per_second,
            "bucket_capacity": g.bucket_capacity,
            "current_tokens": g.tokens,
            "utilization": g.tokens / g.bucket_capacity,
            "total_requests": g.total_requests,
            "rejected_requests": g.rejected_requests,
            "rejection_rate": rejection_rate,
        })
    }
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use nexcore_homeostasis_primitives::enums::CircuitState;
    use tokio::time;

    // ── CircuitBreaker ────────────────────────────────────────────────────────

    #[tokio::test]
    async fn circuit_starts_closed_and_allows_requests() {
        let cb = CircuitBreaker::with_defaults("t");
        assert_eq!(cb.state().await, CircuitState::Closed);
        assert!(cb.allow_request().await);
    }

    #[tokio::test]
    async fn circuit_opens_after_failure_threshold() {
        let cb = CircuitBreaker::new("t", 3, 2, Duration::from_secs(30), 3);
        for _ in 0..3 {
            cb.record_failure().await;
        }
        assert_eq!(cb.state().await, CircuitState::Open);
        assert!(!cb.allow_request().await);
    }

    #[tokio::test]
    async fn circuit_does_not_open_before_threshold() {
        let cb = CircuitBreaker::new("t", 5, 2, Duration::from_secs(30), 3);
        for _ in 0..4 {
            cb.record_failure().await;
        }
        assert_eq!(cb.state().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn success_resets_failure_count() {
        let cb = CircuitBreaker::new("t", 3, 2, Duration::from_secs(30), 3);
        cb.record_failure().await;
        cb.record_failure().await;
        cb.record_success().await; // resets failure count to 0
        cb.record_failure().await;
        cb.record_failure().await;
        // Only 2 failures since last success — threshold is 3, so still closed.
        assert_eq!(cb.state().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn open_transitions_to_half_open_after_timeout() {
        time::pause();
        let cb = CircuitBreaker::new("t", 1, 2, Duration::from_secs(30), 3);
        cb.record_failure().await;
        assert_eq!(cb.state().await, CircuitState::Open);
        time::advance(Duration::from_secs(31)).await;
        // allow_request triggers the auto-transition Open → HalfOpen
        let allowed = cb.allow_request().await;
        assert_eq!(cb.state().await, CircuitState::HalfOpen);
        // In HalfOpen the first request is allowed (half_open_max_requests = 3)
        assert!(allowed);
    }

    #[tokio::test]
    async fn half_open_closes_on_enough_successes() {
        time::pause();
        let cb = CircuitBreaker::new("t", 1, 2, Duration::from_secs(30), 5);
        cb.record_failure().await;
        time::advance(Duration::from_secs(31)).await;
        let _entered = cb.allow_request().await; // triggers HalfOpen
        assert_eq!(cb.state().await, CircuitState::HalfOpen);
        cb.record_success().await;
        cb.record_success().await;
        assert_eq!(cb.state().await, CircuitState::Closed);
    }

    #[tokio::test]
    async fn half_open_reopens_on_failure() {
        time::pause();
        let cb = CircuitBreaker::new("t", 1, 2, Duration::from_secs(30), 5);
        cb.record_failure().await;
        time::advance(Duration::from_secs(31)).await;
        let _entered = cb.allow_request().await;
        cb.record_failure().await;
        assert_eq!(cb.state().await, CircuitState::Open);
    }

    #[tokio::test]
    async fn force_open_blocks_immediately() {
        let cb = CircuitBreaker::with_defaults("t");
        cb.force_open().await;
        assert_eq!(cb.state().await, CircuitState::Open);
        assert!(!cb.allow_request().await);
    }

    #[tokio::test]
    async fn force_close_allows_requests_after_open() {
        let cb = CircuitBreaker::new("t", 1, 2, Duration::from_secs(30), 3);
        cb.record_failure().await;
        assert_eq!(cb.state().await, CircuitState::Open);
        cb.force_close().await;
        assert_eq!(cb.state().await, CircuitState::Closed);
        assert!(cb.allow_request().await);
    }

    #[tokio::test]
    async fn statistics_tracks_total_and_blocked() {
        let cb = CircuitBreaker::new("t", 1, 2, Duration::from_secs(30), 3);
        cb.allow_request().await; // allowed (total=1)
        cb.allow_request().await; // allowed (total=2)
        cb.record_failure().await; // trips
        cb.allow_request().await; // blocked (total=3, blocked=1)
        let stats = cb.get_statistics().await;
        assert_eq!(stats["total_requests"], 3);
        assert_eq!(stats["blocked_requests"], 1);
    }

    // ── RateLimiter ───────────────────────────────────────────────────────────

    #[tokio::test]
    async fn limiter_allows_initial_full_bucket() {
        let rl = RateLimiter::new("t", 10.0, 5.0);
        for _ in 0..5 {
            assert!(rl.acquire().await);
        }
    }

    #[tokio::test]
    async fn limiter_rejects_when_bucket_empty() {
        let rl = RateLimiter::new("t", 0.0, 3.0);
        for _ in 0..3 {
            rl.acquire().await;
        }
        assert!(!rl.acquire().await, "should reject when empty");
    }

    #[tokio::test]
    async fn limiter_refills_over_time() {
        time::pause();
        let rl = RateLimiter::new("t", 10.0, 10.0);
        for _ in 0..10 {
            rl.acquire().await;
        }
        assert!(!rl.acquire().await);
        time::advance(Duration::from_secs(2)).await;
        assert!(rl.acquire().await, "should allow after refill");
    }

    #[tokio::test]
    async fn limiter_set_rate_adjusts_refill_speed() {
        time::pause();
        let rl = RateLimiter::new("t", 100.0, 100.0);
        for _ in 0..100 {
            rl.acquire().await;
        }
        rl.set_rate(1.0).await;
        time::advance(Duration::from_secs(1)).await;
        let tokens = rl.current_tokens().await;
        // Should be ~1 token (±0.1 for float precision).
        assert!(
            tokens >= 0.9 && tokens <= 2.0,
            "expected ~1 token, got {tokens}"
        );
    }

    #[tokio::test]
    async fn limiter_utilization_decreases_as_bucket_drains() {
        let rl = RateLimiter::new("t", 100.0, 100.0);
        let before = rl.utilization().await;
        for _ in 0..50 {
            rl.acquire().await;
        }
        let after = rl.utilization().await;
        assert!(after < before, "utilization should decrease");
    }

    #[tokio::test]
    async fn limiter_statistics_tracks_rejections() {
        let rl = RateLimiter::new("t", 0.0, 2.0);
        rl.acquire().await; // allowed
        rl.acquire().await; // allowed
        rl.acquire().await; // rejected
        let stats = rl.get_statistics().await;
        assert_eq!(stats["rejected_requests"], 1);
        assert_eq!(stats["total_requests"], 3);
    }
}
