# SPDX-License-Identifier: Apache-2.0
"""
Head-mounted Intel RealSense D455 for RBY1 Isaac Sim 5.1
========================================================

What this module does
---------------------
1. Finds `link_head_2` under the RBY1 USD.
2. Creates a fixed mount Xform under that link.
3. References NVIDIA Isaac Sim's RealSense D455 digital-twin USD.
4. Disables the D455 asset's own rigid-body simulation so it follows the head.
5. Creates 640x480 render products for:
     - D455 color camera
     - D455 pseudo-depth camera
6. Publishes ROS 2:
     /rby1/head_camera/color/image_raw
     /rby1/head_camera/color/camera_info
     /rby1/head_camera/depth/image_raw
     /rby1/head_camera/depth/camera_info
-----------
- Isaac Sim 5.1
- isaacsim.ros2.bridge is enabled BEFORE calling setup_head_realsense()
- Simulation domain is ROS_DOMAIN_ID=20
- RBY1 model has exactly one prim named `link_head_2`

The initial mount transform is deliberately easy to tune at the top of this file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import omni.replicator.core as rep
import omni.syntheticdata
import omni.syntheticdata._syntheticdata as sd
import omni.graph.core as og

from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.storage.native import get_assets_root_path, get_full_asset_path
from pxr import Gf, UsdGeom, UsdPhysics


# ---------------------------------------------------------------------------
# User-tunable configuration
# ---------------------------------------------------------------------------

HEAD_LINK_NAME = "link_head_2"

# Starting guess:
# D455's default forward direction works without an extra rotation on NVIDIA's
# reference mobile-robot example.  We therefore keep the camera basically
# forward-facing and pitch it slightly downward.
#
# If the physical model is inside the head or too far out, tune ONLY these.
MOUNT_TRANSLATION_M = (0.05, 0.0, 0.055)

# Local XYZ Euler degrees.  With a conventional +X-forward / +Z-up robot frame,
# +Y pitch points the camera slightly downward.
MOUNT_ROTATION_DEG = (0.0, 0.0, 0.0)

RESOLUTION = (640, 480)

NODE_NAMESPACE = "rby1/head_camera"

COLOR_IMAGE_TOPIC = "color/image_raw"
COLOR_INFO_TOPIC = "color/camera_info"
DEPTH_IMAGE_TOPIC = "depth/image_raw"
DEPTH_INFO_TOPIC = "depth/camera_info"

COLOR_FRAME_ID = "head_camera_color_optical_frame"
DEPTH_FRAME_ID = "head_camera_depth_optical_frame"


# Isaac Sim 5.1 documentation uses the first path below.
# Extra fallbacks make the module tolerant to older asset-package layouts.
D455_ASSET_CANDIDATES = (
    "/Isaac/Sensors/RealSense/D455/rsd455.usd",
    "/Isaac/Sensors/Realsense/D455/rsd455.usd",
    "/Isaac/Sensors/Intel/RealSense/D455/rsd455.usd",
    "/Isaac/Sensors/Intel/RealSense/rsd455.usd",
    "/Isaac/Sensors/intel/RealSense/rsd455.usd",
)


# ---------------------------------------------------------------------------
# Returned handles
# ---------------------------------------------------------------------------

@dataclass
class HeadRealSenseHandles:
    head_path: str
    mount_path: str
    asset_root_path: str
    color_camera_path: str
    depth_camera_path: str
    color_render_product: str
    depth_render_product: str
    writers: list[Any]


# ---------------------------------------------------------------------------
# USD helpers
# ---------------------------------------------------------------------------

def _find_unique_prim_by_name(stage, name: str) -> str:
    matches = [str(prim.GetPath()) for prim in stage.Traverse() if prim.GetName() == name]

    if len(matches) != 1:
        raise RuntimeError(
            f"[HeadRealSense] Expected exactly one prim named {name!r}, "
            f"found {len(matches)}: {matches}"
        )

    return matches[0]


def _resolve_d455_asset() -> str:
    # Prefer Isaac Sim's own resolver because the asset layout changed between
    # releases.  The first candidate is the documented Isaac Sim 5.1 path.
    for candidate in D455_ASSET_CANDIDATES:
        for query in (candidate, candidate.lstrip("/")):
            try:
                resolved = get_full_asset_path(query)
                if resolved:
                    return resolved
            except Exception:
                pass

    assets_root = get_assets_root_path()
    if not assets_root:
        raise RuntimeError("[HeadRealSense] Isaac Sim assets root could not be resolved.")

    # Final 5.1 fallback.
    return assets_root.rstrip("/") + D455_ASSET_CANDIDATES[0]


def _set_mount_transform(stage, mount_path: str) -> None:
    mount = UsdGeom.Xform.Define(stage, mount_path)
    api = UsdGeom.XformCommonAPI(mount.GetPrim())

    api.SetTranslate(Gf.Vec3d(*MOUNT_TRANSLATION_M))
    api.SetRotate(
        Gf.Vec3f(*MOUNT_ROTATION_DEG),
        UsdGeom.XformCommonAPI.RotationOrderXYZ,
    )


def _disable_d455_rigid_body(stage, d455_body_path: str) -> None:
    """
    NVIDIA's D455 asset ships with a rigid body.  When mounting it under a robot
    link we disable that rigid body so the sensor follows its parent link.
    """
    prim = stage.GetPrimAtPath(d455_body_path)

    if not prim.IsValid():
        print(f"[HeadRealSense] WARNING: D455 body prim not yet found: {d455_body_path}")
        return

    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        rigid_api = UsdPhysics.RigidBodyAPI(prim)
        rigid_api.CreateRigidBodyEnabledAttr().Set(False)
        print(f"[HeadRealSense] Disabled sensor rigid body: {d455_body_path}")
    else:
        attr = prim.GetAttribute("physics:rigidBodyEnabled")
        if attr.IsValid():
            attr.Set(False)
            print(f"[HeadRealSense] Disabled sensor rigid body attribute: {d455_body_path}")


def _render_product_path(rp) -> str:
    if isinstance(rp, str):
        return rp
    if hasattr(rp, "path"):
        return str(rp.path)
    return str(rp)


# ---------------------------------------------------------------------------
# ROS 2 writer helpers
# ---------------------------------------------------------------------------

def _get_read_camera_info():
    # Isaac Sim 5.1 docs expose this from isaacsim.ros2.bridge.
    # Newer internal layouts may expose it from isaacsim.ros2.core.
    try:
        from isaacsim.ros2.bridge import read_camera_info
        return read_camera_info
    except ImportError:
        from isaacsim.ros2.core import read_camera_info
        return read_camera_info


def _scaled_camera_info(render_product_path: str, width: int, height: int):
    """
    Read camera calibration and rescale K/P to the actual render resolution.

    This is intentional: in Isaac Sim 5.1, CameraInfo can otherwise reflect
    the USD camera's nominal resolution instead of a lower render-product
    resolution.
    """
    import numpy as np

    read_camera_info = _get_read_camera_info()
    info, _ = read_camera_info(render_product_path=render_product_path)

    src_w = float(info.width)
    src_h = float(info.height)
    sx = float(width) / src_w if src_w > 0 else 1.0
    sy = float(height) / src_h if src_h > 0 else 1.0

    k = np.asarray(info.k, dtype=float).reshape(3, 3).copy()
    r = np.asarray(info.r, dtype=float).reshape(3, 3).copy()
    p = np.asarray(info.p, dtype=float).reshape(3, 4).copy()

    # Scale horizontal intrinsics.
    k[0, 0] *= sx
    k[0, 2] *= sx
    p[0, 0] *= sx
    p[0, 2] *= sx
    p[0, 3] *= sx

    # Scale vertical intrinsics.
    k[1, 1] *= sy
    k[1, 2] *= sy
    p[1, 1] *= sy
    p[1, 2] *= sy
    p[1, 3] *= sy

    return {
        "width": int(width),
        "height": int(height),
        "projection_type": info.distortion_model,
        "k": k,
        "r": r,
        "p": p,
        "distortion_model": info.distortion_model,
        "distortion_coefficients": info.d,
    }


def _publish_image(
    render_product_path: str,
    sensor_type_name: str,
    frame_id: str,
    topic: str,
):
    rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
        sensor_type_name
    )

    writer = rep.writers.get(rv + "ROS2PublishImage")

    writer.initialize(
        frameId=frame_id,
        nodeNamespace=NODE_NAMESPACE,
        queueSize=1,
        topicName=topic,
    )

    writer.attach([render_product_path])

    # IMPORTANT:
    # RBY1 simulation renders at 30 Hz.
    # Trigger this ROS publisher on every render frame.
    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        rv + "IsaacSimulationGate",
        render_product_path,
    )

    og.Controller.attribute(
        gate_path + ".inputs:step"
    ).set(1)

    print(
        f"[HeadRealSense] Image publisher active: "
        f"/{NODE_NAMESPACE}/{topic}"
    )

    return writer


def _publish_camera_info(render_product_path: str, frame_id: str, topic: str):
    width, height = RESOLUTION
    info = _scaled_camera_info(render_product_path, width, height)

    writer = rep.writers.get("ROS2PublishCameraInfo")
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=NODE_NAMESPACE,
        queueSize=1,
        topicName=topic,
        width=info["width"],
        height=info["height"],
        projectionType=info["projection_type"],
        k=info["k"].reshape([1, 9]),
        r=info["r"].reshape([1, 9]),
        p=info["p"].reshape([1, 12]),
        physicalDistortionModel=info["distortion_model"],
        physicalDistortionCoefficients=info["distortion_coefficients"],
    )
    writer.attach([render_product_path])
    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        "PostProcessDispatchIsaacSimulationGate",
        render_product_path,
    )

    og.Controller.attribute(
        gate_path + ".inputs:step"
    ).set(1)

    print(
        f"[HeadRealSense] CameraInfo publisher active: "
        f"/{NODE_NAMESPACE}/{topic}"
    )
    return writer


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def mount_head_realsense(stage, robot_prim_path="/World/RBY1"):
    head_path = _find_unique_prim_by_name(stage, HEAD_LINK_NAME)

    mount_path = f"{head_path}/head_realsense_mount"
    sensor_ref_path = f"{mount_path}/rsd455"

    _set_mount_transform(stage, mount_path)

    asset_path = _resolve_d455_asset()

    print(f"[HeadRealSense] Loading D455 asset: {asset_path}")

    add_reference_to_stage(
        usd_path=asset_path,
        prim_path=sensor_ref_path,
    )

    d455_body_path = f"{sensor_ref_path}/RSD455"

    color_camera_path = (
        f"{d455_body_path}/Camera_OmniVision_OV9782_Color"
    )

    depth_camera_path = (
        f"{d455_body_path}/Camera_Pseudo_Depth"
    )

    _disable_d455_rigid_body(stage, d455_body_path)

    print("[HeadRealSense] Mounted")

    return {
        "head_path": head_path,
        "mount_path": mount_path,
        "color_camera_path": color_camera_path,
        "depth_camera_path": depth_camera_path,
    }

def start_head_realsense_ros(camera_paths):
    color_camera_path = camera_paths["color_camera_path"]
    depth_camera_path = camera_paths["depth_camera_path"]

    color_rp = rep.create.render_product(
        color_camera_path,
        RESOLUTION,
        name="RBY1_Head_D455_Color",
    )

    depth_rp = rep.create.render_product(
        depth_camera_path,
        RESOLUTION,
        name="RBY1_Head_D455_Depth",
    )

    color_rp_path = _render_product_path(color_rp)
    depth_rp_path = _render_product_path(depth_rp)

    writers = [
        _publish_image(
            color_rp_path,
            sd.SensorType.Rgb.name,
            COLOR_FRAME_ID,
            COLOR_IMAGE_TOPIC,
        ),

        _publish_camera_info(
            color_rp_path,
            COLOR_FRAME_ID,
            COLOR_INFO_TOPIC,
        ),

        _publish_image(
            depth_rp_path,
            sd.SensorType.DistanceToImagePlane.name,
            DEPTH_FRAME_ID,
            DEPTH_IMAGE_TOPIC,
        ),

        _publish_camera_info(
            depth_rp_path,
            DEPTH_FRAME_ID,
            DEPTH_INFO_TOPIC,
        ),
    ]

    print("[HeadRealSense] ROS publishers READY")

    return writers