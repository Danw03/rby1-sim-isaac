# SPDX-License-Identifier: Apache-2.0
"""
Right-EE D405-like RGB-D camera for RBY1 Isaac Sim 5.1.

Call order:
    mount_ee_right_realsense(stage, robot_prim_path)
        -> during RBY1Task.set_up_scene()

    start_ee_right_realsense_ros(handles)
        -> after world.reset() and renderer initialization

IMPORTANT:
Import this module only after SimulationApp has been instantiated.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import omni.graph.core as og
import omni.replicator.core as rep
import omni.syntheticdata
import omni.syntheticdata._syntheticdata as sd

from pxr import Gf, UsdGeom


# -----------------------------------------------------------------------------
# User-tunable wrist mount
# -----------------------------------------------------------------------------

EE_LINK_NAME = "ee_right"

# Same transform as the current simulation/mock camera TF:
# translation = [0, 0, 0]
# quaternion xyzw = [0, +sqrt(1/2), 0, +sqrt(1/2)]
MOUNT_TRANSLATION_M = (0.0, 0.0, 0.0)
MOUNT_ROTATION_XYZW = (
    0.0,
    0.7071067811865476,
    0.0,
    0.7071067811865476,
)


# -----------------------------------------------------------------------------
# D405-like profile
# -----------------------------------------------------------------------------

RESOLUTION = (640, 360)
HFOV_DEG = 87.0

COLOR_CLIP_M = (0.05, 2.00)
DEPTH_CLIP_M = (0.07, 0.50)

NODE_NAMESPACE = "rby1/right_camera"

COLOR_IMAGE_TOPIC = "color/image_raw"
COLOR_INFO_TOPIC = "color/camera_info"
DEPTH_IMAGE_TOPIC = "depth/image_raw"
DEPTH_INFO_TOPIC = "depth/camera_info"

COLOR_FRAME_ID = "right_d405_color_optical_frame"
DEPTH_FRAME_ID = "right_d405_depth_optical_frame"

# visual-only housing approximation
HOUSING_SIZE_M = (0.042, 0.023, 0.042)


@dataclass
class EERightRealSenseHandles:
    ee_path: str
    camera_link_path: str
    color_camera_path: str
    depth_camera_path: str
    color_render_product: str | None = None
    depth_render_product: str | None = None
    writers: list[Any] | None = None


def _find_unique_prim_by_name(stage, name: str) -> str:
    matches = [
        str(prim.GetPath())
        for prim in stage.Traverse()
        if prim.GetName() == name
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"[EERightCamera] Expected exactly one prim named {name!r}, "
            f"found {len(matches)}: {matches}"
        )
    return matches[0]


def _configure_pinhole(
    stage,
    camera_path: str,
    *,
    hfov_deg: float,
    resolution: tuple[int, int],
    clip_m: tuple[float, float],
) -> None:
    prim = stage.GetPrimAtPath(camera_path)
    if not prim.IsValid() or not prim.IsA(UsdGeom.Camera):
        raise RuntimeError(f"[EERightCamera] Invalid camera prim: {camera_path}")

    camera = UsdGeom.Camera(prim)

    horizontal_aperture = 20.0
    focal_length = horizontal_aperture / (
        2.0 * math.tan(math.radians(hfov_deg) / 2.0)
    )
    aspect = float(resolution[0]) / float(resolution[1])
    vertical_aperture = horizontal_aperture / aspect

    camera.GetHorizontalApertureAttr().Set(horizontal_aperture)
    camera.GetVerticalApertureAttr().Set(vertical_aperture)
    camera.GetFocalLengthAttr().Set(focal_length)
    camera.GetClippingRangeAttr().Set(Gf.Vec2f(*clip_m))


def _make_camera(stage, path: str, clip_m: tuple[float, float]):
    camera = UsdGeom.Camera.Define(stage, path)

    # USD camera looks down local -Z.  Rotate it so the D405 camera-link +X
    # acts as sensor-forward.
    api = UsdGeom.XformCommonAPI(camera.GetPrim())
    api.SetRotate(
        Gf.Vec3f(90.0, 0.0, -90.0),
        UsdGeom.XformCommonAPI.RotationOrderXYZ,
    )

    _configure_pinhole(
        stage,
        path,
        hfov_deg=HFOV_DEG,
        resolution=RESOLUTION,
        clip_m=clip_m,
    )

    return camera


def _render_product_path(render_product) -> str:
    if isinstance(render_product, str):
        return render_product
    if hasattr(render_product, "path"):
        return str(render_product.path)
    return str(render_product)


def _get_read_camera_info():
    try:
        from isaacsim.ros2.bridge import read_camera_info
        return read_camera_info
    except ImportError:
        from isaacsim.ros2.core import read_camera_info
        return read_camera_info


def _scaled_camera_info(render_product_path: str):
    import numpy as np

    info, _ = _get_read_camera_info()(render_product_path=render_product_path)

    width, height = RESOLUTION
    src_w = float(info.width)
    src_h = float(info.height)

    sx = float(width) / src_w if src_w > 0.0 else 1.0
    sy = float(height) / src_h if src_h > 0.0 else 1.0

    k = np.asarray(info.k, dtype=float).reshape(3, 3).copy()
    r = np.asarray(info.r, dtype=float).reshape(3, 3).copy()
    p = np.asarray(info.p, dtype=float).reshape(3, 4).copy()

    k[0, 0] *= sx
    k[0, 2] *= sx
    k[1, 1] *= sy
    k[1, 2] *= sy

    p[0, 0] *= sx
    p[0, 2] *= sx
    p[0, 3] *= sx
    p[1, 1] *= sy
    p[1, 2] *= sy
    p[1, 3] *= sy

    model = info.distortion_model or "plumb_bob"

    return {
        "width": int(width),
        "height": int(height),
        "projection_type": model,
        "k": k,
        "r": r,
        "p": p,
        "distortion_model": model,
        "distortion_coefficients": info.d,
    }


def _publish_image(
    render_product_path: str,
    *,
    frame_id: str,
    topic: str,
    sensor_type_name: str,
):
    render_var = (
        omni.syntheticdata.SyntheticData
        .convert_sensor_type_to_rendervar(sensor_type_name)
    )

    writer = rep.writers.get(render_var + "ROS2PublishImage")
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=NODE_NAMESPACE,
        queueSize=1,
        topicName=topic,
    )
    writer.attach([render_product_path])

    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        render_var + "IsaacSimulationGate",
        render_product_path,
    )
    og.Controller.attribute(gate_path + ".inputs:step").set(1)

    return writer


def _publish_camera_info(render_product_path: str, *, frame_id: str, topic: str):
    info = _scaled_camera_info(render_product_path)

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
    og.Controller.attribute(gate_path + ".inputs:step").set(1)

    return writer


def mount_ee_right_realsense(
    stage,
    robot_prim_path: str = "/World/RBY1",
) -> EERightRealSenseHandles:
    """Mount a D405-like RGB-D camera under ee_right."""
    ee_path = _find_unique_prim_by_name(stage, EE_LINK_NAME)

    if not ee_path.startswith(robot_prim_path.rstrip("/") + "/"):
        raise RuntimeError(
            f"[EERightCamera] Found {EE_LINK_NAME} at {ee_path}, "
            f"expected under {robot_prim_path}"
        )

    camera_link_path = f"{ee_path}/right_d405_link"

    old = stage.GetPrimAtPath(camera_link_path)
    if old.IsValid():
        stage.RemovePrim(camera_link_path)

    root = UsdGeom.Xform.Define(stage, camera_link_path)
    xformable = UsdGeom.Xformable(root.GetPrim())

    xformable.AddTranslateOp().Set(
        Gf.Vec3d(*MOUNT_TRANSLATION_M)
    )

    qx, qy, qz, qw = (float(v) for v in MOUNT_ROTATION_XYZW)
    norm = math.sqrt(qx*qx + qy*qy + qz*qz + qw*qw)
    if norm <= 1.0e-12:
        raise ValueError("[EERightCamera] Mount quaternion must be non-zero.")

    qx, qy, qz, qw = (v / norm for v in (qx, qy, qz, qw))

    xformable.AddOrientOp().Set(
        Gf.Quatd(qw, Gf.Vec3d(qx, qy, qz))
    )

    # Visual-only D405-like enclosure.
    housing = UsdGeom.Cube.Define(stage, f"{camera_link_path}/Housing")
    housing.CreateSizeAttr(1.0)
    housing.AddScaleOp().Set(Gf.Vec3f(*HOUSING_SIZE_M))
    housing.CreateDisplayColorAttr([Gf.Vec3f(0.12, 0.12, 0.14)])

    color_path = f"{camera_link_path}/ColorCamera"
    depth_path = f"{camera_link_path}/DepthCamera"

    _make_camera(stage, color_path, COLOR_CLIP_M)
    _make_camera(stage, depth_path, DEPTH_CLIP_M)

    print(
        "[EERightCamera] D405-like camera mounted | "
        f"translation={MOUNT_TRANSLATION_M} | "
        f"rotation_xyzw={MOUNT_ROTATION_XYZW} | "
        f"{RESOLUTION[0]}x{RESOLUTION[1]} | "
        f"depth={DEPTH_CLIP_M[0]:.2f}-{DEPTH_CLIP_M[1]:.2f}m"
    )

    return EERightRealSenseHandles(
        ee_path=ee_path,
        camera_link_path=camera_link_path,
        color_camera_path=color_path,
        depth_camera_path=depth_path,
    )


def start_ee_right_realsense_ros(
    handles: EERightRealSenseHandles,
) -> EERightRealSenseHandles:
    """Start ROS 2 publishers for the right-wrist RGB-D camera."""
    color_rp = rep.create.render_product(
        handles.color_camera_path,
        RESOLUTION,
        name="RBY1_Right_D405_Color",
    )
    depth_rp = rep.create.render_product(
        handles.depth_camera_path,
        RESOLUTION,
        name="RBY1_Right_D405_Depth",
    )

    color_rp_path = _render_product_path(color_rp)
    depth_rp_path = _render_product_path(depth_rp)

    writers = [
        _publish_image(
            color_rp_path,
            frame_id=COLOR_FRAME_ID,
            topic=COLOR_IMAGE_TOPIC,
            sensor_type_name=sd.SensorType.Rgb.name,
        ),
        _publish_camera_info(
            color_rp_path,
            frame_id=COLOR_FRAME_ID,
            topic=COLOR_INFO_TOPIC,
        ),
        _publish_image(
            depth_rp_path,
            frame_id=DEPTH_FRAME_ID,
            topic=DEPTH_IMAGE_TOPIC,
            sensor_type_name=sd.SensorType.DistanceToImagePlane.name,
        ),
        _publish_camera_info(
            depth_rp_path,
            frame_id=DEPTH_FRAME_ID,
            topic=DEPTH_INFO_TOPIC,
        ),
    ]

    handles.color_render_product = color_rp_path
    handles.depth_render_product = depth_rp_path
    handles.writers = writers

    print("[EERightCamera] ROS publishers READY")
    print(f"  /{NODE_NAMESPACE}/{COLOR_IMAGE_TOPIC}")
    print(f"  /{NODE_NAMESPACE}/{COLOR_INFO_TOPIC}")
    print(f"  /{NODE_NAMESPACE}/{DEPTH_IMAGE_TOPIC}")
    print(f"  /{NODE_NAMESPACE}/{DEPTH_INFO_TOPIC}")

    return handles
