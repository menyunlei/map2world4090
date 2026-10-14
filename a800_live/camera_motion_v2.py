"""Bounded live camera controls; native camera anchor timing is unchanged.

This helper changes only requested camera motion. It creates no RGB frames,
changes no model state, and has no dependency on torch or a GPU.
"""
from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class MotionConfigV2:
    max_speed_mps: float = 2.0
    max_yaw_rate_degrees_per_second: float = 35.0
    ramp_seconds: float = 0.30
    integration_hz: int = 240

    def __post_init__(self):
        for value in (self.max_speed_mps,
                      self.max_yaw_rate_degrees_per_second,
                      self.ramp_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Motion limits must be finite and positive")
        if not isinstance(self.integration_hz, int) or self.integration_hz <= 0:
            raise ValueError("integration_hz must be a positive integer")


def _towards(current, target, maximum_change):
    return current + max(-maximum_change, min(maximum_change, target - current))


def _rate_integral(current, target, acceleration, duration):
    """Integral of an acceleration-limited rate, including target arrival."""
    if duration <= 0:
        return 0.0
    difference = target - current
    ramp_time = min(duration, abs(difference) / acceleration)
    signed_acceleration = math.copysign(acceleration, difference) if difference else 0.0
    return (current * ramp_time + 0.5 * signed_acceleration * ramp_time ** 2
            + target * (duration - ramp_time))


class CameraMotionV2:
    """Unicycle camera in X/Z with independently bounded speed and yaw rate.

    Call command(keys) once when the native three-latent chunk samples input.
    Call advance_to(pixel_time) at each original native camera anchor. The
    returned position/yaw can be used directly to construct the original C2W
    matrix. A change from rest to a maximum rate, or maximum to neutral, takes
    0.30 s. A full forward/reverse or left/right reversal takes 0.60 s.
    """

    def __init__(self, config=None):
        self.config = config or MotionConfigV2()
        self.time = 0.0
        self.x = self.z = self.yaw = 0.0
        self.speed = self.yaw_rate = 0.0
        self.target_speed = self.target_yaw_rate = 0.0
        self.keys = ""
        self.max_yaw_rate = math.radians(self.config.max_yaw_rate_degrees_per_second)
        self.speed_acceleration = self.config.max_speed_mps / self.config.ramp_seconds
        self.yaw_acceleration = self.max_yaw_rate / self.config.ramp_seconds

    def command(self, keys):
        """Opposing keys cancel; releasing a key requests a bounded stop."""
        self.keys = "".join(sorted(set(str(keys).lower()) & set("wasd")))
        self.target_speed = self.config.max_speed_mps * (
            int("w" in self.keys) - int("s" in self.keys))
        self.target_yaw_rate = self.max_yaw_rate * (
            int("d" in self.keys) - int("a" in self.keys))
        return self.snapshot()

    def _step(self, duration):
        speed_before, yaw_rate_before, yaw_before = self.speed, self.yaw_rate, self.yaw
        # Two-point Gaussian quadrature for the continuous unicycle path;
        # yaw itself is integrated analytically. Target arrivals are step
        # boundaries in advance_to, so each rate is linear over this step.
        for fraction in (0.5 - math.sqrt(3.0) / 6.0,
                         0.5 + math.sqrt(3.0) / 6.0):
            elapsed = duration * fraction
            speed = _towards(speed_before, self.target_speed,
                             self.speed_acceleration * elapsed)
            heading = yaw_before + _rate_integral(
                yaw_rate_before, self.target_yaw_rate, self.yaw_acceleration, elapsed)
            self.x += 0.5 * duration * speed * math.sin(heading)
            self.z += 0.5 * duration * speed * math.cos(heading)
        self.yaw += _rate_integral(yaw_rate_before, self.target_yaw_rate,
                                  self.yaw_acceleration, duration)
        self.speed = _towards(speed_before, self.target_speed,
                              self.speed_acceleration * duration)
        self.yaw_rate = _towards(yaw_rate_before, self.target_yaw_rate,
                                 self.yaw_acceleration * duration)

    def advance_to(self, pixel_time):
        """Advance to the unmodified native pixel-camera time, in seconds."""
        pixel_time = float(pixel_time)
        if not math.isfinite(pixel_time) or pixel_time < self.time - 1e-12:
            raise ValueError("Camera time must be finite and monotonic")
        while pixel_time - self.time > 1e-12:
            # Global grid rather than a caller-relative subdivision keeps
            # native-anchor results invariant to native chunk partitioning.
            next_grid = (math.floor(self.time * self.config.integration_hz + 1e-9)
                         + 1) / self.config.integration_hz
            duration = min(pixel_time - self.time, next_grid - self.time)
            for current, target, acceleration in (
                    (self.speed, self.target_speed, self.speed_acceleration),
                    (self.yaw_rate, self.target_yaw_rate, self.yaw_acceleration)):
                remaining = abs(target - current) / acceleration
                if remaining > 1e-12:
                    duration = min(duration, remaining)
            if duration <= 1e-12:
                # Snap a floating-point grid boundary without making a
                # zero-duration loop or changing the camera endpoint.
                self.time = min(pixel_time, next_grid)
                continue
            self._step(duration)
            self.time += duration
        self.time = max(self.time, pixel_time)
        return self.snapshot()

    def snapshot(self):
        return {
            "camera_position": [self.x, 0.0, self.z],
            "camera_yaw_radians": self.yaw,
            "camera_yaw_degrees": math.degrees(self.yaw),
            "camera_time_seconds": self.time,
            "keys": self.keys,
            "target_speed_mps": self.target_speed,
            "effective_speed_mps": self.speed,
            "target_yaw_rate_degrees_per_second": math.degrees(self.target_yaw_rate),
            "effective_yaw_rate_degrees_per_second": math.degrees(self.yaw_rate),
        }

    def contract(self):
        return dict(asdict(self.config),
                    speed_acceleration_mps2=self.speed_acceleration,
                    yaw_acceleration_degrees_per_second2=math.degrees(self.yaw_acceleration),
                    full_direction_reversal_seconds=2.0 * self.config.ramp_seconds,
                    changes_camera_conditioning_only=True,
                    native_anchor_timing_unchanged=True,
                    native_chunk_input_sampling_unchanged=True,
                    frame_interpolation=False)
