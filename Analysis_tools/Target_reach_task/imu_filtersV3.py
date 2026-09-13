"""
imu_filters.py
Pluggable orientation and position estimation filters for comparing phone
dead-reckoning accuracy in arm_eazy's evaluation chapter.

Orientation filters (roll/pitch only - yaw needs a magnetometer, out of
scope here since a 2D tilt-to-cursor mapping doesn't need it):
    - naive          : raw gyro integration, no correction (baseline / worst case)
    - complementary   : cheap gyro + accel-tilt blend
    - madgwick        : gradient-descent quaternion fusion (via ahrs package)
    - mahony          : PI-controller quaternion fusion (via ahrs package)

Position filters (2D, per-axis independent):
    - naive           : raw double integration of acceleration (baseline / worst case)
    - zupt            : naive + threshold-based zero-velocity reset
    - kalman_zupt     : Kalman filter (pos, vel) with accel as control input and
                        zero-velocity update as a proper measurement correction
    - kalman_zupt_bias: as above, with an added per-axis acceleration-bias state
                        (candidate refinement motivated by the axis-asymmetric
                        drift seen in the stationary translation stability test)
    - glrt_zupt       : as kalman_zupt, but stillness is decided by a windowed
                        Generalized-Likelihood-Ratio-Test style statistic
                        (accel variance + gyro variance over a short sliding
                        window) rather than an instantaneous magnitude
                        threshold -- distinguishes a genuine sustained stop
                        from a brief near-zero crossing at a motion reversal

Default gains below were revisited after the sensor stability characterization
(stationary vs. in-hand, 300s @ ~20 Hz, Google Pixel 9a) rather than left as
library/arbitrary defaults:
    - Madgwick beta   : derived from measured in-hand gyro noise std via the
                        heuristic beta ~= sqrt(3/4) * sigma_gyro
    - ZUPT window     : reduced from 0.25s to 0.15s, since live reach-task
                        logs showed sustained-stillness periods topping out
                        at ~0.217s -- the original window never triggered
                        during actual task use
    - Kalman Q, R     : re-derived from measured in-hand (Q) and stationary
                        (R) linear-acceleration variance instead of
                        hand-picked constants
Gyroscope bias subtraction was evaluated and NOT implemented: measured
per-axis bias (~1e-6 rad/s) was ~3 orders of magnitude smaller than the
gyro's own noise std (~1e-3 rad/s), so naive-integration drift is a
noise-driven random walk rather than a correctable fixed offset.

Requirements:
    pip install ahrs numpy
"""

import math

import numpy as np

# ---------------------------------------------------------------------------
# Calibration-derived constants (see docstring above / thesis Section [X] for
# the stability-test data these were derived from)
# ---------------------------------------------------------------------------

# In-hand gyroscope noise std, rad/s (per-axis range observed: ~0.025-0.038;
# midpoint used here). Drives the Madgwick beta heuristic below.
MEASURED_GYRO_STD_INHAND = 0.03
DEFAULT_MADGWICK_BETA = math.sqrt(0.75) * MEASURED_GYRO_STD_INHAND  # ~0.026

# Linear-acceleration variance, (m/s^2)^2 -- in-hand (process noise driver)
# and stationary (zero-velocity measurement noise driver).
MEASURED_ACCEL_VAR_INHAND = 0.005
MEASURED_ACCEL_VAR_STATIONARY = 2e-5

# GLRT stillness-detector calibration (see move/stop/move/stop and mov_x/mov_y
# validation logs referenced in thesis Section [X]):
#   - window must beat the shortest genuine still phase observed (~0.9s) while
#     comfortably clearing the longest spurious near-zero crossing seen during
#     continuous oscillatory motion (~0.23s) -- 0.3s sits between the two.
#   - accel/gyro variance thresholds sit roughly an order of magnitude above
#     the stationary noise floor and an order of magnitude below the in-hand
#     noise floor for each signal, so genuine stillness and genuine motion are
#     both cleanly separated from the decision boundary.
GLRT_WINDOW_S = 0.3
GLRT_ACCEL_VAR_THRESH = 1e-3
GLRT_GYRO_VAR_THRESH = 5e-5

# Velocity deadband, m/s -- applied AFTER a position filter's step(), not
# inside any one filter, so it's consistent across naive/zupt/kalman_zupt/
# glrt_zupt alike. Calibrated from a live glrt_zupt task run: residual
# velocity during confirmed ZUPT-active (stationary) periods reaches up to
# ~0.014-0.021 m/s (p90-p99), while genuine intentional motion is mostly
# above ~0.025 m/s (p25). 0.02 sits between the two, absorbing steady-state
# Kalman-correction residual/noise creep without suppressing real motion.
# NOTE: this is device/session-specific noise, not a universal constant --
# re-check these percentiles if the gain, filter, or device changes.
VELOCITY_DEADBAND = 0.02


def apply_velocity_deadband(vx, vy, threshold=VELOCITY_DEADBAND):
    """Zero out (vx, vy) if their combined magnitude is below threshold.
    Use on a position filter's step() output to suppress small residual
    velocity (e.g. steady-state Kalman-ZUPT correction noise) that would
    otherwise integrate into visible cursor creep, without needing an
    oversized threshold that also suppresses genuine slow motion.

    NOTE: a plain deadband like this can still produce visible "backlash" --
    a small cursor bounce-back right after stopping or reversing. That
    happens when a genuine (not noise) natural hand deceleration slightly
    overshoots past zero velocity at the end of a reach: the small reversal
    clears the threshold in the OPPOSITE direction and gets let straight
    through. VelocityHysteresis below addresses this specifically; prefer
    it over this plain deadband if backlash is the symptom you're seeing."""
    if math.hypot(vx, vy) < threshold:
        return 0.0, 0.0
    return vx, vy


# Hysteresis release threshold, m/s -- how far velocity must climb above
# VELOCITY_DEADBAND, once latched at zero, before being let through again.
# Like DEFAULT_VELOCITY_DAMPING, this is a control-feel trade-off, not a
# measured noise-floor constant: too small and small overshoot-reversals
# ("backlash") still sneak through; too large and genuine gentle direction
# changes get suppressed/delayed. 3x the latch threshold is a starting
# point -- tune by feel.
VELOCITY_HYSTERESIS_RELEASE = VELOCITY_DEADBAND * 3


class VelocityHysteresis:
    """Schmitt-trigger-style hysteresis on velocity, applied per axis, to
    eliminate "backlash" -- a small cursor bounce-back right after stopping
    or reversing direction. A plain magnitude deadband (VELOCITY_DEADBAND /
    apply_velocity_deadband) can't fix this: it clamps small values to zero
    but lets the very next small value back through immediately, including
    a small reversal in the opposite direction -- which is exactly what
    produces the visible bounce, since natural hand deceleration at the end
    of a reach often overshoots slightly past true zero velocity before
    settling (a real signal, not sensor noise).

    Unlike apply_velocity_deadband (a stateless function), this needs to
    remember whether each axis is currently "latched" at zero, so it's a
    class with per-session state rather than a plain function -- construct
    ONE instance per task run and reuse it every step, not one per call.
    """

    def __init__(self, latch_threshold=VELOCITY_DEADBAND,
                 release_threshold=VELOCITY_HYSTERESIS_RELEASE):
        self.latch_threshold = latch_threshold
        self.release_threshold = release_threshold
        self._latched_x = False
        self._latched_y = False

    def _apply_axis(self, v, latched):
        if latched:
            if abs(v) < self.release_threshold:
                return 0.0, True  # stay latched -- suppress small reversal
            return v, False  # cleared the higher bar -- genuine motion, release
        if abs(v) < self.latch_threshold:
            return 0.0, True  # just settled -- latch at zero
        return v, False  # already moving -- pass through unchanged

    def apply(self, vx, vy):
        vx, self._latched_x = self._apply_axis(vx, self._latched_x)
        vy, self._latched_y = self._apply_axis(vy, self._latched_y)
        return vx, vy


# Minimum duration, s, that velocity must stay on ONE side of
# VELOCITY_HYSTERESIS_RELEASE before DirectionLock releases it. A jerk
# spike (the deceleration transient at the end of a reach) crosses a
# magnitude threshold for a sample or two and then drops back -- it does
# NOT sustain. Genuine reversed motion keeps growing/holding. This is what
# actually distinguishes "jerk" from "real reversal": duration, not size --
# VelocityHysteresis alone still releases on a single sample that happens
# to clear the threshold, which is why a large-enough jerk can still slip
# through hysteresis and read as a direction change. 0.08s is a starting
# point -- shorter feels snappier but lets shorter jerks through; longer
# suppresses jerks more reliably but adds perceptible lag before a
# genuine quick direction change registers. Tune by feel.
DEFAULT_CONFIRM_DURATION = 0.08


class DirectionLock:
    """Direction-confirmation lock: distinguishes a genuine sustained
    reversal from a brief jerk-driven reversal spike ("backlash") by
    requiring the candidate new direction to not just clear a magnitude
    threshold once (as VelocityHysteresis does) but to hold continuously,
    same sign, for confirm_duration before being released. The moment the
    signal drops back below threshold OR flips sign again, the confirm
    timer resets to zero -- so a single jerky sample, however large, can
    never accumulate enough sustained duration to release on its own.

    While a candidate direction is being confirmed, output stays at zero
    (same as while fully latched) -- this trades a small amount of added
    latency (up to confirm_duration) on a genuine direction change for
    eliminating jerk-driven bounce-back. Per-axis, stateful: construct ONE
    instance per task run, call apply() every step with the real dt.
    """

    def __init__(self, latch_threshold=VELOCITY_DEADBAND,
                 release_threshold=VELOCITY_HYSTERESIS_RELEASE,
                 confirm_duration=DEFAULT_CONFIRM_DURATION):
        self.latch_threshold = latch_threshold
        self.release_threshold = release_threshold
        self.confirm_duration = confirm_duration
        self._latched = [True, True]        # per axis: x, y
        self._confirm_t = [0.0, 0.0]         # time accumulated on current candidate sign
        self._confirm_sign = [0, 0]          # +1 / -1 / 0 (no candidate yet)

    def _apply_axis(self, i, v, dt):
        if self._latched[i]:
            if abs(v) < self.release_threshold:
                # dropped back below the bar -- reset any partial confirm
                self._confirm_t[i] = 0.0
                self._confirm_sign[i] = 0
                return 0.0
            sign = 1 if v > 0 else -1
            if sign == self._confirm_sign[i]:
                self._confirm_t[i] += dt
            else:
                # sign changed mid-confirm (e.g. jerk then reversed jerk) --
                # start the confirm window over for the new sign
                self._confirm_sign[i] = sign
                self._confirm_t[i] = dt
            if self._confirm_t[i] >= self.confirm_duration:
                self._latched[i] = False  # sustained long enough -- release
                return v
            return 0.0  # still confirming -- suppress for now
        else:
            if abs(v) < self.latch_threshold:
                self._latched[i] = True
                self._confirm_t[i] = 0.0
                self._confirm_sign[i] = 0
                return 0.0
            return v

    def apply(self, vx, vy, dt):
        vx = self._apply_axis(0, vx, dt)
        vy = self._apply_axis(1, vy, dt)
        return vx, vy


# Acceleration deadband, m/s^2 -- applied to raw accel BEFORE it reaches a
# position filter's step(), so noise is trimmed at the source rather than
# only after it has already been integrated into velocity.
#
# Unlike velocity, raw acceleration does NOT cleanly separate stationary
# from moving by magnitude alone: a live glrt_zupt run showed stationary
# (GLRT-confirmed still) accel magnitude reaching p90=0.236, p95=0.281 m/s^2
# -- higher than the MEDIAN of genuine motion (0.140). This is expected
# (it's exactly why GLRT judges stillness from variance over a window, not
# a single sample) but it means a magnitude threshold can't fully suppress
# stationary noise without also cutting real slow motion.
#
# ACCEL_DEADBAND is therefore set conservatively -- below genuine motion's
# ~p5-p10 (0.024-0.055) -- to trim only the smallest jitter. It will NOT
# remove most stationary-noise spikes; that's left to GLRT/ZUPT (which
# corrects velocity, not raw accel) and VELOCITY_DEADBAND downstream.
# Raising this further would start suppressing genuine gentle motion.
ACCEL_DEADBAND = 0.03


def apply_accel_deadband(ax, ay, threshold=ACCEL_DEADBAND):
    """Zero out (ax, ay) if their combined magnitude is below threshold.
    Deliberately small (see ACCEL_DEADBAND comment) -- this only trims the
    smallest sensor jitter, it is not a stillness detector on its own."""
    if math.hypot(ax, ay) < threshold:
        return 0.0, 0.0
    return ax, ay


# Velocity damping ("friction"), 1/s -- decays velocity toward zero every
# step regardless of whether ZUPT/GLRT has confirmed stillness yet. This is
# a DIFFERENT fix from VELOCITY_DEADBAND: the deadband only hides small
# residual velocity once it's already small; it does nothing about genuine
# velocity coasting at constant speed between real stillness corrections
# (the Kalman predict step assumes constant velocity when accel is ~0, so
# without damping, nothing pulls velocity down until a full GLRT/ZUPT
# window has elapsed -- which is exactly the "drifts too much before
# stopping" symptom).
#
# This is a control-feel / UX parameter, not a measured noise-floor
# constant like the ones above -- there's no "correct" value from data,
# only a trade-off: higher damping stops the cursor faster but also bleeds
# off genuine sustained motion faster (making deliberate steady movement
# feel like it's fighting friction); lower damping preserves momentum but
# drifts longer after release. 2.0 (~0.5s time constant: velocity decays to
# ~37% after 0.5s, ~14% after 1s with no new accel input) is a starting
# point -- tune by feel, ideally exposed as a CLI flag rather than hardcoded.
DEFAULT_VELOCITY_DAMPING = 0.02


def _decay(v, damping, dt):
    return v * max(0.0, 1.0 - damping * dt)


# ---------------------------------------------------------------------------
# Orientation filters
# ---------------------------------------------------------------------------

class NaiveGyroOrientation:
    name = "naive"

    def __init__(self):
        self.roll = 0.0
        self.pitch = 0.0

    def update(self, gx, gy, gz, ax, ay, az, dt):
        self.roll += gx * dt
        self.pitch += gy * dt
        return self.roll, self.pitch


class ComplementaryOrientation:
    name = "complementary"

    def __init__(self, alpha=0.98):
        self.alpha = alpha
        self.roll = 0.0
        self.pitch = 0.0

    def update(self, gx, gy, gz, ax, ay, az, dt):
        roll_acc = math.atan2(ay, az) if (ay or az) else 0.0
        pitch_acc = math.atan2(-ax, math.hypot(ay, az)) if (ay or az or ax) else 0.0

        roll_gyro = self.roll + gx * dt
        pitch_gyro = self.pitch + gy * dt

        self.roll = self.alpha * roll_gyro + (1 - self.alpha) * roll_acc
        self.pitch = self.alpha * pitch_gyro + (1 - self.alpha) * pitch_acc
        return self.roll, self.pitch


class _AhrsQuaternionOrientation:
    """Shared plumbing for ahrs-package filters (Madgwick, Mahony)."""

    def __init__(self, ahrs_filter):
        self.filt = ahrs_filter
        self.q = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z

    def update(self, gx, gy, gz, ax, ay, az, dt):
        gyr = np.array([gx, gy, gz])
        acc = np.array([ax, ay, az])
        dt = max(dt, 1e-4)
        self.q = self.filt.updateIMU(self.q, gyr=gyr, acc=acc, dt=dt)
        roll, pitch = self._roll_pitch_from_quat(self.q)
        return roll, pitch

    @staticmethod
    def _roll_pitch_from_quat(q):
        w, x, y, z = q
        # standard quaternion -> Euler (roll = X rotation, pitch = Y rotation)
        roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
        sinp = 2 * (w * y - z * x)
        sinp = max(-1.0, min(1.0, sinp))
        pitch = math.asin(sinp)
        return roll, pitch


class MadgwickOrientation(_AhrsQuaternionOrientation):
    name = "madgwick"

    def __init__(self, beta=DEFAULT_MADGWICK_BETA):
        # beta default derived from measured in-hand gyro noise (see module
        # constants above), replacing the ahrs library default of 0.1
        from ahrs.filters import Madgwick
        super().__init__(Madgwick(gain=beta))


class MahonyOrientation(_AhrsQuaternionOrientation):
    name = "mahony"

    def __init__(self, kp=1.0, ki=0.01):
        from ahrs.filters import Mahony
        super().__init__(Mahony(k_P=kp, k_I=ki))


ORIENTATION_FILTERS = {
    "naive": NaiveGyroOrientation,
    "complementary": ComplementaryOrientation,
    "madgwick": MadgwickOrientation,
    "mahony": MahonyOrientation,
}


# ---------------------------------------------------------------------------
# Position filters (2D dead reckoning from linear acceleration)
# ---------------------------------------------------------------------------

class NaivePosition:
    name = "naive"

    def __init__(self):
        self.vx = 0.0
        self.vy = 0.0

    def step(self, ax, ay, dt):
        self.vx += ax * dt
        self.vy += ay * dt
        return self.vx, self.vy, False  # (vel_x, vel_y, zupt_active)


class ZUPTPosition:
    name = "zupt"

    def __init__(self, threshold=0.15, window=0.15, damping=DEFAULT_VELOCITY_DAMPING):
        # window reduced from the original 0.25s: live reach-task logs never
        # sustained stillness past ~0.217s, so 0.25s meant ZUPT never fired
        # during actual task use. threshold unchanged -- stationary noise
        # ceiling (~0.03-0.06 m/s^2) stays well clear of 0.15 m/s^2.
        # damping: see DEFAULT_VELOCITY_DAMPING comment -- decays velocity
        # every step so the cursor slows down before ZUPT confirms
        # stillness, rather than coasting at constant speed until it does.
        self.threshold = threshold
        self.window = window
        self.damping = damping
        self.vx = 0.0
        self.vy = 0.0
        self._low_since = None
        self._t = 0.0

    def step(self, ax, ay, dt):
        self._t += dt
        mag = math.hypot(ax, ay)
        active = False
        if mag < self.threshold:
            if self._low_since is None:
                self._low_since = self._t
            elif self._t - self._low_since >= self.window:
                self.vx, self.vy = 0.0, 0.0
                active = True
        else:
            self._low_since = None

        self.vx = _decay(self.vx, self.damping, dt) + ax * dt
        self.vy = _decay(self.vy, self.damping, dt) + ay * dt
        return self.vx, self.vy, active


class KalmanZUPTPosition:
    """Per-axis 2-state (pos, vel) Kalman filter. Acceleration is the
    control input; a zero-velocity pseudo-measurement is fused whenever
    the phone is judged stationary (low sustained accel magnitude)."""

    name = "kalman_zupt"

    def __init__(self, threshold=0.15, window=0.15,
                 process_noise=MEASURED_ACCEL_VAR_INHAND,
                 zupt_noise=MEASURED_ACCEL_VAR_STATIONARY,
                 damping=DEFAULT_VELOCITY_DAMPING):
        # process_noise/zupt_noise now derived from measured in-hand /
        # stationary linear-acceleration variance rather than hand-picked
        # constants (previously 0.5 / 1e-4); window reduced to match the
        # ZUPTPosition change above, same task-motion-profile rationale.
        # damping: see DEFAULT_VELOCITY_DAMPING comment -- applied to the
        # velocity state in the predict step so it decays every sample,
        # not just once ZUPT confirms stillness.
        self.threshold = threshold
        self.window = window
        self.q = process_noise
        self.r = zupt_noise
        self.damping = damping
        self._low_since = None
        self._t = 0.0

        # state [pos, vel] and covariance, one per axis
        self.x = np.zeros((2, 2))       # rows: x-axis, y-axis ; cols: pos, vel
        self.P = np.array([np.eye(2) * 1.0, np.eye(2) * 1.0])

    def _predict_axis(self, state, P, a, dt):
        F = np.array([[1, dt], [0, 1]])
        B = np.array([0.5 * dt ** 2, dt])
        Q = np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]]) * self.q
        state = F @ state + B * a
        state[1] = _decay(state[1], self.damping, dt)
        P = F @ P @ F.T + Q
        return state, P

    def _zupt_update_axis(self, state, P):
        H = np.array([0.0, 1.0])
        R = self.r
        y = 0.0 - (H @ state)
        S = H @ P @ H.T + R
        K = (P @ H) / S
        state = state + K * y
        P = P - np.outer(K, H) @ P
        return state, P

    def step(self, ax, ay, dt):
        self._t += dt
        mag = math.hypot(ax, ay)
        active = False
        if mag < self.threshold:
            if self._low_since is None:
                self._low_since = self._t
            elif self._t - self._low_since >= self.window:
                active = True
        else:
            self._low_since = None

        for i, a in enumerate((ax, ay)):
            self.x[i], self.P[i] = self._predict_axis(self.x[i], self.P[i], a, dt)
            if active:
                self.x[i], self.P[i] = self._zupt_update_axis(self.x[i], self.P[i])

        vx, vy = self.x[0, 1], self.x[1, 1]
        return vx, vy, active


class KalmanZUPTBiasPosition:
    """Per-axis 3-state (pos, vel, bias) Kalman filter -- candidate
    refinement over KalmanZUPTPosition. Motivated by the stationary
    translation stability test, where naive double-integration drift was
    far larger and axis-asymmetric (e.g. one axis ~10x the others),
    consistent with a small residual constant acceleration bias compounding
    quadratically under double integration rather than pure zero-mean noise.

    The bias state is included in the prediction model and is left
    uncorrected by the zero-velocity update (H only observes velocity), so
    it must be identifiable from the process model alone over time; this is
    a structural change from KalmanZUPTPosition, not just a parameter
    change, and is offered here as an optional/experimental variant pending
    further validation rather than a drop-in replacement."""

    name = "kalman_zupt_bias"

    def __init__(self, threshold=0.15, window=0.15,
                 process_noise=MEASURED_ACCEL_VAR_INHAND,
                 zupt_noise=MEASURED_ACCEL_VAR_STATIONARY,
                 bias_process_noise=1e-6,
                 damping=DEFAULT_VELOCITY_DAMPING):
        self.threshold = threshold
        self.window = window
        self.q = process_noise
        self.r = zupt_noise
        self.qb = bias_process_noise
        self.damping = damping
        self._low_since = None
        self._t = 0.0

        # state [pos, vel, bias] and covariance, one per axis
        self.x = np.zeros((2, 3))
        self.P = np.array([np.eye(3) * 1.0, np.eye(3) * 1.0])

    def _predict_axis(self, state, P, a, dt):
        # bias is modeled as a slowly-varying additive term on measured
        # acceleration: a_true = a_meas - bias
        F = np.array([[1, dt, 0],
                      [0, 1, -dt],
                      [0, 0, 1]])
        B = np.array([0.5 * dt ** 2, dt, 0.0])
        Q = np.diag([dt ** 4 / 4 * self.q, dt ** 2 * self.q, self.qb])
        state = F @ state + B * a
        state[1] = _decay(state[1], self.damping, dt)
        P = F @ P @ F.T + Q
        return state, P

    def _zupt_update_axis(self, state, P):
        H = np.array([0.0, 1.0, 0.0])
        R = self.r
        y = 0.0 - (H @ state)
        S = H @ P @ H.T + R
        K = (P @ H) / S
        state = state + K * y
        P = P - np.outer(K, H) @ P
        return state, P

    def step(self, ax, ay, dt):
        self._t += dt
        mag = math.hypot(ax, ay)
        active = False
        if mag < self.threshold:
            if self._low_since is None:
                self._low_since = self._t
            elif self._t - self._low_since >= self.window:
                active = True
        else:
            self._low_since = None

        for i, a in enumerate((ax, ay)):
            self.x[i], self.P[i] = self._predict_axis(self.x[i], self.P[i], a, dt)
            if active:
                self.x[i], self.P[i] = self._zupt_update_axis(self.x[i], self.P[i])

        vx, vy = self.x[0, 1], self.x[1, 1]
        return vx, vy, active


class GLRTStillnessDetector:
    """Windowed Generalized-Likelihood-Ratio-Test-style stillness detector.

    Instead of an instantaneous magnitude threshold on acceleration alone
    (ZUPTPosition, KalmanZUPTPosition), this maintains a short sliding window
    of accelerometer AND gyroscope samples and declares "stationary" only
    when BOTH signals show low variance over the window. Requiring both
    signals to agree, over a window rather than a single sample, is what
    lets this distinguish a genuine sustained stop from a brief near-zero
    acceleration crossing at a motion reversal (which the accelerometer
    alone can't tell apart from a real stop, but which the gyroscope -
    still showing rotational motion from the ongoing reach - can).

    This is a practical simplification of the windowed test statistic used
    in foot-mounted INS / pedestrian dead-reckoning literature (e.g. Skog et
    al.), which combines accel and gyro variance into a single likelihood
    ratio against a reference model; here the two variances are checked
    against independently calibrated thresholds rather than combined into a
    single closed-form statistic, trading some statistical elegance for a
    simpler, directly-tunable implementation.
    """

    def __init__(self, window_s=GLRT_WINDOW_S,
                 accel_var_thresh=GLRT_ACCEL_VAR_THRESH,
                 gyro_var_thresh=GLRT_GYRO_VAR_THRESH):
        self.window_s = window_s
        self.accel_var_thresh = accel_var_thresh
        self.gyro_var_thresh = gyro_var_thresh
        self._buf = []  # list of (t, ax, ay, gx, gy, gz)
        self._t = 0.0

    def update(self, ax, ay, gx, gy, gz, dt):
        self._t += dt
        self._buf.append((self._t, ax, ay, gx, gy, gz))
        cutoff = self._t - self.window_s
        while self._buf and self._buf[0][0] < cutoff:
            self._buf.pop(0)

        if len(self._buf) < 3:
            return False  # not enough samples yet to judge

        arr = np.array(self._buf)
        accel = arr[:, 1:3]
        gyro = arr[:, 3:6]
        accel_var = accel.var(axis=0).sum()
        gyro_var = gyro.var(axis=0).sum()
        return bool(accel_var < self.accel_var_thresh and gyro_var < self.gyro_var_thresh)


class GLRTZUPTPosition:
    """As KalmanZUPTPosition, but stillness is decided by a
    GLRTStillnessDetector (accel+gyro windowed variance test) instead of an
    instantaneous magnitude threshold. Requires gyro readings alongside
    acceleration at each step, unlike the other position filters."""

    name = "glrt_zupt"
    needs_gyro = True  # step() requires gx, gy, gz in addition to ax, ay, dt

    def __init__(self, window_s=GLRT_WINDOW_S,
                 accel_var_thresh=GLRT_ACCEL_VAR_THRESH,
                 gyro_var_thresh=GLRT_GYRO_VAR_THRESH,
                 process_noise=MEASURED_ACCEL_VAR_INHAND,
                 zupt_noise=MEASURED_ACCEL_VAR_STATIONARY,
                 damping=DEFAULT_VELOCITY_DAMPING):
        self.detector = GLRTStillnessDetector(window_s, accel_var_thresh, gyro_var_thresh)
        self.q = process_noise
        self.r = zupt_noise
        self.damping = damping

        # state [pos, vel] and covariance, one per axis (same structure as
        # KalmanZUPTPosition)
        self.x = np.zeros((2, 2))
        self.P = np.array([np.eye(2) * 1.0, np.eye(2) * 1.0])

    def _predict_axis(self, state, P, a, dt):
        F = np.array([[1, dt], [0, 1]])
        B = np.array([0.5 * dt ** 2, dt])
        Q = np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt ** 2]]) * self.q
        state = F @ state + B * a
        state[1] = _decay(state[1], self.damping, dt)
        P = F @ P @ F.T + Q
        return state, P

    def _zupt_update_axis(self, state, P):
        H = np.array([0.0, 1.0])
        R = self.r
        y = 0.0 - (H @ state)
        S = H @ P @ H.T + R
        K = (P @ H) / S
        state = state + K * y
        P = P - np.outer(K, H) @ P
        return state, P

    def step(self, ax, ay, gx, gy, gz, dt):
        active = self.detector.update(ax, ay, gx, gy, gz, dt)

        for i, a in enumerate((ax, ay)):
            self.x[i], self.P[i] = self._predict_axis(self.x[i], self.P[i], a, dt)
            if active:
                self.x[i], self.P[i] = self._zupt_update_axis(self.x[i], self.P[i])

        vx, vy = self.x[0, 1], self.x[1, 1]
        return vx, vy, active


POSITION_FILTERS = {
    "naive": NaivePosition,
    "zupt": ZUPTPosition,
    "kalman_zupt": KalmanZUPTPosition,
    "kalman_zupt_bias": KalmanZUPTBiasPosition,
    "glrt_zupt": GLRTZUPTPosition,
}