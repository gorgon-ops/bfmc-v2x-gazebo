# What the filter is doing, and how well. Published alongside every estimate
# so a run can be judged after the fact without re-deriving anything.
std_msgs/Header header

float32 position_error      # m from Gazebo ground truth (simulation only)
float32 heading_error       # rad from ground truth
float32 dead_reckoning_error  # m, the same vehicle without any V2X fixes

float32 position_sigma      # m, the filter's own 1-sigma estimate
float32 yaw_sigma           # rad
float32 gyro_bias           # rad/s, estimated

float32 last_nis            # normalised innovation squared of the last fix
uint32 accepted_fixes
uint32 rejected_fixes       # gated out as implausible
uint32 stale_fixes          # older than the replay buffer
uint32 replayed_events      # cumulative cost of the delayed-fusion path
float32 fix_age             # s, latency of the most recent fix
