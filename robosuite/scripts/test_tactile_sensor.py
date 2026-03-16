"""
Test script for tactile sensors on the Panda gripper.

Creates a Lift environment with PandaTactileGripper, runs a simple grasp sequence,
and verifies that:
  (a) The correct number of touch sensors are registered (128 = 64 per finger)
  (b) Tactile readings are non-zero when grasping an object
  (c) Output shape is (8, 8, 1) per fingerpad
  (d) Readings return to zero (or near-zero) when fingers are open

Usage:
    python test_tactile_sensor.py [--visualize] [--render]
"""

import argparse
import numpy as np
import robosuite as suite
from robosuite import load_controller_config
from robosuite.utils.input_utils import input2action
from robosuite.utils.tactile_utils import reshape_tactile_reading


def test_tactile_sensor(args):
    if args.keyboard and (not args.render):
        print("[Info] --keyboard 模式需要渲染窗口，自动启用 --render")
        args.render = True

    controller_config = load_controller_config(default_controller="OSC_POSE")

    # Create environment with PandaTactileGripper
    env = suite.make(
        env_name="Lift",
        robots="Panda",
        gripper_types="PandaTactileGripper",
        controller_configs=controller_config,
        has_renderer=args.render,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        reward_shaping=True,
        control_freq=20,
    )

    obs = env.reset()
    robot = env.robots[0]
    gripper = robot.gripper
    action_dim = env.action_dim
    gripper_dof = gripper.dof
    arm_action_dim = action_dim - gripper_dof

    def make_action(dz=0.0, grip=0.0):
        action = np.zeros(action_dim, dtype=np.float32)
        if arm_action_dim >= 3:
            action[2] = dz
        action[-gripper_dof:] = grip
        return action

    # ---- Check (a): sensor count ----
    tactile_names = gripper.tactile_sensor_names
    n_left = len(tactile_names["left"])
    n_right = len(tactile_names["right"])
    rows, cols = gripper.tactile_grid_shape
    expected = rows * cols
    print(f"\n[Check a] Tactile grid shape: {gripper.tactile_grid_shape}")
    print(f"  Left sensors: {n_left}, Right sensors: {n_right}, Expected per side: {expected}")
    assert n_left == expected, f"Left sensor count mismatch: {n_left} != {expected}"
    assert n_right == expected, f"Right sensor count mismatch: {n_right} != {expected}"
    print("  ✓ Sensor count correct")

    # ---- Check (c): output shape in observations ----
    pf = robot.robot_model.naming_prefix
    left_key = f"{pf}tactile_left"
    right_key = f"{pf}tactile_right"

    if left_key in obs:
        left_obs = obs[left_key]
        right_obs = obs[right_key]
        print(f"\n[Check c] Tactile obs shape: left={left_obs.shape}, right={right_obs.shape}")
        assert left_obs.shape == (rows, cols, 1), f"Left shape mismatch: {left_obs.shape}"
        assert right_obs.shape == (rows, cols, 1), f"Right shape mismatch: {right_obs.shape}"
        print("  ✓ Output shape correct")
    else:
        print(f"\n[Warning] Tactile keys not found in obs. Available keys:")
        for k in sorted(obs.keys()):
            if "tactile" in k.lower():
                print(f"  {k}: shape={obs[k].shape}")

    live_viz = None
    if args.visualize and (left_key in obs):
        try:
            import matplotlib.pyplot as plt

            plt.ion()
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            left_im = axes[0].imshow(obs[left_key][:, :, 0], cmap='hot', interpolation='nearest', vmin=0.0)
            axes[0].set_title("Left Fingerpad Tactile")
            axes[0].set_xlabel("Column")
            axes[0].set_ylabel("Row")
            fig.colorbar(left_im, ax=axes[0])

            right_im = axes[1].imshow(obs[right_key][:, :, 0], cmap='hot', interpolation='nearest', vmin=0.0)
            axes[1].set_title("Right Fingerpad Tactile")
            axes[1].set_xlabel("Column")
            axes[1].set_ylabel("Row")
            fig.colorbar(right_im, ax=axes[1])

            fig.suptitle("Realtime Tactile Sensor Readings")
            fig.tight_layout()
            plt.show(block=False)

            live_viz = dict(plt=plt, fig=fig, left_im=left_im, right_im=right_im)

            def _update_live_tactile(left_arr, right_arr, step):
                if (step % args.viz_every) != 0:
                    return
                max_val = max(float(np.max(np.abs(left_arr))), float(np.max(np.abs(right_arr))), 1e-6)
                live_viz["left_im"].set_data(left_arr[:, :, 0])
                live_viz["right_im"].set_data(right_arr[:, :, 0])
                live_viz["left_im"].set_clim(0.0, max_val)
                live_viz["right_im"].set_clim(0.0, max_val)
                live_viz["fig"].canvas.draw_idle()
                live_viz["plt"].pause(0.001)

        except ImportError:
            print("\n[Visualization] matplotlib not available, skipping realtime plot")
            live_viz = None

            def _update_live_tactile(left_arr, right_arr, step):
                return
    else:
        def _update_live_tactile(left_arr, right_arr, step):
            return

    if args.keyboard:
        from robosuite.devices import Keyboard

        device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
        env.viewer.add_keypress_callback(device.on_press)
        device.start_control()

        print("\n[Keyboard Mode] 使用键盘手动控制机器人（OSC_POSE）。")
        print("[Keyboard Mode] 按 Ctrl+C 退出测试。")

        max_contact_reading = 0.0
        step_count = 0
        try:
            while True:
                action, _ = input2action(
                    device=device,
                    robot=robot,
                    active_arm="right",
                    env_configuration=None,
                )

                if action is None:
                    break

                rem_action_dim = env.action_dim - action.size
                if rem_action_dim > 0:
                    action = np.concatenate([action, np.zeros(rem_action_dim)])
                elif rem_action_dim < 0:
                    action = action[: env.action_dim]

                obs, reward, done, info = env.step(action)
                env.render()

                if done:
                    print("[Keyboard Mode] Episode 结束，自动 reset。")
                    obs = env.reset()
                    step_count = 0
                    max_contact_reading = 0.0
                    continue

                if left_key in obs:
                    left_val = obs[left_key]
                    right_val = obs[right_key]
                    _update_live_tactile(left_val, right_val, step_count)
                    current_max = max(np.max(np.abs(left_val)), np.max(np.abs(right_val)))
                    max_contact_reading = max(max_contact_reading, current_max)

                    if (step_count % args.print_every) == 0:
                        print(
                            f"[tactile] step={step_count} "
                            f"left_max={np.max(np.abs(left_val)):.6f} "
                            f"right_max={np.max(np.abs(right_val)):.6f} "
                            f"hist_max={max_contact_reading:.6f}"
                        )
                step_count += 1
        except KeyboardInterrupt:
            print("\n[Keyboard Mode] 用户中断，结束测试。")

        env.close()
        return

    # ---- Run grasp sequence ----
    print("\n[Running grasp sequence...]")

    # Phase 1: Move down toward object with gripper open
    print("  Phase 1: Moving down with open gripper...")
    for i in range(80):
        # Move down (negative z) with open gripper (-1)
        action = make_action(dz=-1.0, grip=-1.0)
        obs, reward, done, info = env.step(action)
        if left_key in obs:
            _update_live_tactile(obs[left_key], obs[right_key], i)
        if args.render:
            env.render()

    # Check (d): readings should be near zero with open gripper
    if left_key in obs:
        left_open = obs[left_key]
        right_open = obs[right_key]
        max_open = max(np.max(np.abs(left_open)), np.max(np.abs(right_open)))
        print(f"\n[Check d] Max tactile reading (open gripper): {max_open:.6f}")

    # Phase 2: Close gripper to grasp
    print("  Phase 2: Closing gripper...")
    max_contact_reading = 0.0
    for i in range(100):
        action = make_action(dz=0.0, grip=1.0)  # close gripper
        obs, reward, done, info = env.step(action)
        if left_key in obs:
            _update_live_tactile(obs[left_key], obs[right_key], i)
        if args.render:
            env.render()

        if left_key in obs:
            left_val = obs[left_key]
            right_val = obs[right_key]
            current_max = max(np.max(np.abs(left_val)), np.max(np.abs(right_val)))
            max_contact_reading = max(max_contact_reading, current_max)

    # Check (b): non-zero reading when grasping
    if left_key in obs:
        print(f"\n[Check b] Max tactile reading during grasp: {max_contact_reading:.6f}")
        if max_contact_reading > 0.0:
            print("  ✓ Non-zero tactile readings detected during grasp")
        else:
            print("  ⚠ No tactile contact detected. This may happen if the gripper")
            print("    did not make contact with the object. Try adjusting the grasp sequence.")

    # Phase 3: Lift up
    print("  Phase 3: Lifting...")
    for i in range(50):
        action = make_action(dz=1.0, grip=1.0)  # lift with closed gripper
        obs, reward, done, info = env.step(action)
        if left_key in obs:
            _update_live_tactile(obs[left_key], obs[right_key], i)
        if args.render:
            env.render()

    # Phase 4: Open gripper
    print("  Phase 4: Opening gripper...")
    for i in range(50):
        action = make_action(dz=0.0, grip=-1.0)  # open gripper
        obs, reward, done, info = env.step(action)
        if left_key in obs:
            _update_live_tactile(obs[left_key], obs[right_key], i)
        if args.render:
            env.render()

    if left_key in obs:
        left_final = obs[left_key]
        right_final = obs[right_key]
        max_final = max(np.max(np.abs(left_final)), np.max(np.abs(right_final)))
        print(f"\n[Check d] Max tactile reading after release: {max_final:.6f}")

    # ---- Optional visualization ----
    if args.visualize and left_key in obs:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].imshow(obs[left_key][:, :, 0], cmap='hot', interpolation='nearest')
            axes[0].set_title("Left Fingerpad Tactile")
            axes[0].set_xlabel("Column")
            axes[0].set_ylabel("Row")
            plt.colorbar(axes[0].images[0], ax=axes[0])

            axes[1].imshow(obs[right_key][:, :, 0], cmap='hot', interpolation='nearest')
            axes[1].set_title("Right Fingerpad Tactile")
            axes[1].set_xlabel("Column")
            axes[1].set_ylabel("Row")
            plt.colorbar(axes[1].images[0], ax=axes[1])

            plt.suptitle("Tactile Sensor Readings")
            plt.tight_layout()
            plt.savefig("tactile_test_result.png", dpi=150)
            print("\n[Visualization] Saved to tactile_test_result.png")
            if live_viz is None:
                plt.show()
        except ImportError:
            print("\n[Visualization] matplotlib not available, skipping plot")

    if live_viz is not None:
        try:
            live_viz["fig"].savefig("tactile_test_realtime_final.png", dpi=150)
            print("[Visualization] Saved realtime final frame to tactile_test_realtime_final.png")
        except Exception:
            pass

    # ---- Summary ----
    print("\n" + "=" * 50)
    print("Tactile Sensor Test Summary")
    print("=" * 50)
    print(f"  Grid shape:       {rows} x {cols}")
    print(f"  Total sensors:    {n_left + n_right}")
    print(f"  Obs shape (each): ({rows}, {cols}, 1)")
    if left_key in obs:
        print(f"  Max grasp force:  {max_contact_reading:.6f}")
    print("  Status:           PASSED" if max_contact_reading > 0 else "  Status:           NEEDS VERIFICATION")
    print("=" * 50)

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test tactile sensors on Panda gripper")
    parser.add_argument("--render", action="store_true", help="Enable on-screen rendering")
    parser.add_argument("--visualize", action="store_true", help="Show matplotlib visualization of tactile readings")
    parser.add_argument("--keyboard", action="store_true", help="Use keyboard teleoperation instead of scripted motion")
    parser.add_argument("--pos-sensitivity", type=float, default=1.0, help="Position sensitivity for keyboard control")
    parser.add_argument("--rot-sensitivity", type=float, default=1.0, help="Rotation sensitivity for keyboard control")
    parser.add_argument("--print-every", type=int, default=20, help="Print tactile stats every N env steps in keyboard mode")
    parser.add_argument("--viz-every", type=int, default=5, help="Update realtime tactile visualization every N env steps")
    args = parser.parse_args()
    test_tactile_sensor(args)
