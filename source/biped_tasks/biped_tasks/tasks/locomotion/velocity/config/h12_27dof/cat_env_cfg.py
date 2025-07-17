# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
    EventTermCfg as EventTerm,
    RewardTermCfg as RewTerm,
    SceneEntityCfg,
    TerminationTermCfg as DoneTerm,
)

from biped_tasks.utils.history.manager_term_cfg import (
    ObservationGroupCfg as ObsGroup,
    ObservationTermCfg as ObsTerm,
)

from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp

from biped_tasks.utils.cat.manager_constraint_cfg import ConstraintTermCfg as ConstraintTerm
import biped_tasks.utils.cat.constraints as constraints
import biped_tasks.utils.cat.curriculums as curriculums
import biped_tasks.utils.mdp.commands as commands
import biped_tasks.utils.mdp.events as events
import biped_tasks.utils.mdp.rewards as rewards

from biped_assets.robots.h12 import H12_27DOF as ROBOT_CFG  # isort: skip
from biped_tasks.utils.mdp.terrains import ROUGH_TERRAINS_CFG  # isort: skip


VELOCITY_DEADZONE = 0.2
MAX_CURRICULUM_ITERATIONS = 5000


# ========================================================
# Scene Configuration
# ========================================================
@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="exts/cat_envs/cat_envs/assets/materials/Shingles_01.mdl",
            project_uvw=True,
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = ROBOT_CFG.replace(
        prim_path="/World/envs/env_.*/Robot"
    )
    # sensors
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )
    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"exts/cat_envs/cat_envs/assets/materials/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


# ========================================================
# Commands Configuration
# ========================================================
@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    base_velocity = commands.UniformVelocityCommandWithDeadzoneCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 8.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=False,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0)
        ),
        velocity_deadzone=VELOCITY_DEADZONE
    )


# ========================================================
# Actions Configuration
# ========================================================
@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "left_hip_yaw_joint",
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_yaw_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
        ],
        scale=0.5,
        use_default_offset=True,
        preserve_order=True,
    )
    
    fixed_joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            "torso_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_yaw_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_yaw_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint"
        ],
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )
    

# ========================================================
# Observations Configuration
# ========================================================
@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        # observation terms (order preserved)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, 
                            noise=Unoise(n_min=-0.01, n_max=0.01), 
                            params={"asset_cfg": SceneEntityCfg("robot", 
                                                                joint_names=[
                                                                    "left_hip_yaw_joint",
                                                                    "left_hip_pitch_joint",
                                                                    "left_hip_roll_joint",
                                                                    "left_knee_joint",
                                                                    "left_ankle_pitch_joint",
                                                                    "left_ankle_roll_joint",
                                                                    "right_hip_yaw_joint",
                                                                    "right_hip_pitch_joint",
                                                                    "right_hip_roll_joint",
                                                                    "right_knee_joint",
                                                                    "right_ankle_pitch_joint",
                                                                    "right_ankle_roll_joint",
                                                                ],
                                                                preserve_order=True)})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel,
                            scale=0.05, 
                            noise=Unoise(n_min=-1.5, n_max=1.5), 
                            params={"asset_cfg": SceneEntityCfg("robot", 
                                                                joint_names=[
                                                                    "left_hip_yaw_joint",
                                                                    "left_hip_pitch_joint",
                                                                    "left_hip_roll_joint",
                                                                    "left_knee_joint",
                                                                    "left_ankle_pitch_joint",
                                                                    "left_ankle_roll_joint",
                                                                    "right_hip_yaw_joint",
                                                                    "right_hip_pitch_joint",
                                                                    "right_hip_roll_joint",
                                                                    "right_knee_joint",
                                                                    "right_ankle_pitch_joint",
                                                                    "right_ankle_roll_joint",
                                                                ],
                                                                preserve_order=True)})
        actions = ObsTerm(func=mdp.last_action,
                          params={"action_name": "joint_pos"})
       
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = 6
            self.history_step = 1

    # observation groups
    policy: PolicyCfg = PolicyCfg()


# ========================================================
# Events Configuration
# ========================================================
@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.1, 1.25),
            "dynamic_friction_range": (0.1, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
        },
    )
    scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.9, 1.2),
            "operation": "scale",
            "recompute_inertia": False,
        },
    )
    # move_base_com = EventTerm(
    #     func=events.randomize_body_coms,
    #     mode="startup",
    #     params={
    #         "max_displacement": 0.05,
    #         "asset_cfg": SceneEntityCfg("robot", body_names="torso_link"),
    #     },
    # )
    # randomize_actuator_gains = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "stiffness_distribution_params": (0.8, 1.2),
    #         "damping_distribution_params": (0.8, 1.2),
    #         "operation": "scale"
    #     },
    # )

    # reset
    base_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*torso_link"),
            "force_range": (0.0, 0.0),
            "torque_range": (-0.0, 0.0),
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.0, 0.0),
                "y": (-0.0, 0.0),
                "z": (-0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),
            "velocity_range": (1.0, 1.0),
        },
    )
    
    # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(5.0, 8.0),
        params={"velocity_range": {"x": (-1.0, 1.0), 
                                   "y": (-1.0, 1.0)}},
    )


# ========================================================
# Rewards Configuration
# ========================================================
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # -- task
    track_lin_vel_xy_exp = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    track_ang_vel_z_exp = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=0.5,
        params={"command_name": "base_velocity", "std": math.sqrt(0.25)},
    )
    
    # -- penalties
    action_rate_l2 = RewTerm(
        func=rewards.action_rate_l2, 
        weight=-0.01,
        params={"asset_cfg": SceneEntityCfg("robot")}
    )


# ========================================================
# Constraints Configuration
# ========================================================
@configclass
class ConstraintsCfg:
    # Safety Hard constraints
    contact = ConstraintTerm(
        func=constraints.contact,
        max_p=1.0,
        params={
            "asset_cfg": SceneEntityCfg("contact_forces", body_names=[".*_hip_yaw_link", ".*_hip_roll_link", ".*_hip_pitch_link", ".*_knee_link", "torso_link", "pelvis", ".*_shoulder_pitch_link", ".*_shoulder_roll_link", ".*_shoulder_yaw_link", ".*_elbow_link", ".*_wrist_yaw_link", ".*_wrist_roll_link", ".*_wrist_pitch_link"])
            }
    )

    # Safety Soft constraints
    # joint_position_limits = ConstraintTerm(
    #     func=constraints.joint_position_limits,
    #     max_p=0.25,
    #     params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
    #         "left_hip_yaw_joint",
    #         "left_hip_pitch_joint",
    #         "left_hip_roll_joint",
    #         "left_knee_joint",
    #         "left_ankle_pitch_joint",
    #         "left_ankle_roll_joint",
    #         "right_hip_yaw_joint",
    #         "right_hip_pitch_joint",
    #         "right_hip_roll_joint",
    #         "right_knee_joint",
    #         "right_ankle_pitch_joint",
    #         "right_ankle_roll_joint",
    #     ])},                       
    # )
    joint_velocity_limits = ConstraintTerm(
        func=constraints.joint_velocity_limits,
        max_p=0.25,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
            "left_hip_yaw_joint",
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_yaw_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
        ])},                       
    )
    joint_torque_limits = ConstraintTerm(
        func=constraints.joint_torque_limits,
        max_p=0.25,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[
            "left_hip_yaw_joint",
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_yaw_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
        ])},                       
    )
    foot_contact_force = ConstraintTerm(
        func=constraints.foot_contact_force,
        max_p=0.25,
        params={
            "limit": 1000.0, 
            "asset_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"])},
    )
    action_rate = ConstraintTerm(
        func=constraints.action_rate,
        max_p=0.25,
        params={"limit": 90.0, 
                "asset_cfg": SceneEntityCfg("robot")},                       
    )
    
    # Style constraints
    base_orientation = ConstraintTerm(
        func=constraints.base_orientation, 
        max_p=0.25, 
        params={"limit": 0.2,
            "asset_cfg": SceneEntityCfg("robot")
                }
    )
    foot_contact = ConstraintTerm(
        func=constraints.foot_contact,
        max_p=0.25,
        params={
            "asset_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"])
                }
    )
    air_time = ConstraintTerm(
        func=constraints.air_time,
        max_p=0.25,
        params={"limit": 0.4, 
                "asset_cfg": SceneEntityCfg("contact_forces", body_names=[".*_ankle_roll_link"]),
                "velocity_deadzone": VELOCITY_DEADZONE,},
    )
    no_move = ConstraintTerm(
        func=constraints.no_move,
        max_p=1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[
            "left_hip_yaw_joint",
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_yaw_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
        ]),                       
            "velocity_deadzone": VELOCITY_DEADZONE,
            "joint_vel_limit": 0.0,
        },
    )
    knee_position = ConstraintTerm(
        func=constraints.joint_range,
        max_p=0.25,
        params={"limit": 0.3, 
                "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_knee_joint"])},                
    )
    left_foot_orientation = ConstraintTerm(
        func=constraints.foot_orientation, 
        max_p=0.25, 
        params={
            "limit": 0.05,
            "desired_projected_gravity": [0.0, 0.0, -1.0],
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_ankle_roll_link"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_ankle_roll_link"])}
    )
    right_foot_orientation = ConstraintTerm(
        func=constraints.foot_orientation, 
        max_p=0.25, 
        params={
            "limit": 0.05,
            "desired_projected_gravity": [0.0, 0.0, -1.0],
            "asset_cfg": SceneEntityCfg("robot", body_names=["right_ankle_roll_link"]),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["right_ankle_roll_link"])}
    )


# ========================================================
# Terminations Configuration
# ========================================================
@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[
                                             ".*_hip_yaw_link",
                                             ".*_hip_roll_link",
                                             ".*_hip_pitch_link",
                                             ".*_knee_link",
                                             "torso_link",
                                             "pelvis",
                                             ".*_shoulder_pitch_link",
                                             ".*_shoulder_roll_link",
                                             ".*_shoulder_yaw_link",
                                             ".*_elbow_link",
                                             ".*_wrist_yaw_link",
                                             ".*_wrist_roll_link",
                                             ".*_wrist_pitch_link"]
            ),
            "threshold": 1.0,
        },
    )


# ========================================================
# Curriculum Configuration
# ========================================================
@configclass
class CurriculumCfg:
    # Soft constraints
    # joint_position_limits = CurrTerm(
    #     func=curriculums.modify_constraint_p,
    #     params={"term_name": "joint_position_limits", 
    #             "num_steps": 24 * MAX_CURRICULUM_ITERATIONS, 
    #             "init_max_p": 0.25},
    # )
    joint_velocity_limits = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={"term_name": "joint_velocity_limits", 
                "num_steps": 24 * MAX_CURRICULUM_ITERATIONS, 
                "init_max_p": 0.25},
    )
    joint_torque_limits = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={"term_name": "joint_torque_limits", 
                "num_steps": 24 * MAX_CURRICULUM_ITERATIONS, 
                "init_max_p": 0.25},
    )
    foot_contact_force = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={"term_name": "foot_contact_force", 
                "num_steps": 24 * MAX_CURRICULUM_ITERATIONS, 
                "init_max_p": 0.25},
    )
    action_rate = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={"term_name": "action_rate", 
                "num_steps": 24 * MAX_CURRICULUM_ITERATIONS, 
                "init_max_p": 0.25},
    )
    
    # Style constraints
    base_orientation = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "base_orientation",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    foot_contact = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={"term_name": "foot_contact", 
                "num_steps": 24 * MAX_CURRICULUM_ITERATIONS, 
                "init_max_p": 0.25},
    )
    air_time = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={"term_name": "air_time", 
                "num_steps": 24 * MAX_CURRICULUM_ITERATIONS, 
                "init_max_p": 0.25},
    )
    no_move = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "no_move",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 1.0,
        },
    )
    knee_position = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "knee_position",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    left_foot_orientation = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "left_foot_orientation",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    right_foot_orientation = CurrTerm(
        func=curriculums.modify_constraint_p,
        params={
            "term_name": "right_foot_orientation",
            "num_steps": 24 * MAX_CURRICULUM_ITERATIONS,
            "init_max_p": 0.25,
        },
    )
    

# ========================================================
# Environment Configuration
# ========================================================
@configclass
class H12_27dof_EnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    constraints: ConstraintsCfg = ConstraintsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 8
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.0025
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = True
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.solver_type = 1
        self.sim.max_position_iteration_count = 4
        self.sim.max_velocity_iteration_count = 0
        self.sim.bounce_threshold_velocity = 0.5
        self.sim.gpu_max_rigid_contact_count = 2**23
        
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt
        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False


class H12_27dof_EnvCfg_PLAY(H12_27dof_EnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 100
        self.scene.env_spacing = 2.5
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False
        
        # self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0., 0.)
        # self.commands.base_velocity.ranges.ang_vel_z = (-.0, .0)