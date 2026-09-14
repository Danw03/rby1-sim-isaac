# SPDX-License-Identifier: Apache-2.0
"""
RBY1 VSLAM lab environment
==========================

Geometry based on the requested floor plan:
- Room: 5.0 m x 10.0 m
- Robot workspace: upper 3.5 m
- Human workspace: lower 6.5 m
- Existing RBY1 spawn can remain at (0, 0)
- Table 1 / Table 2 are L-shaped and stay inside the robot workspace
- Robot workspace floor is intentionally kept obstacle-free
- Human workspace contains static office-like clutter only
- No moving human actor yet

Coordinate convention
---------------------
+X : right
+Y : toward the top wall in the sketch
Robot spawn: (0, 0)

Room:
    X = [-2.5,  2.5]
    Y = [-8.25, 1.75]

Robot workspace:
    Y = [-1.75, 1.75]  (3.5 m)

Human workspace:
    Y = [-8.25, -1.75] (6.5 m)

Call:
    from lab_environment import add_lab_environment
    add_lab_environment(scene.stage)
"""

from __future__ import annotations

from pxr import Gf, UsdGeom, UsdLux, UsdPhysics

from lab_pick_plate import add_pick_plate


# ---------------------------------------------------------------------------
# Main dimensions
# ---------------------------------------------------------------------------

ROOM_WIDTH = 5.0
ROOM_LENGTH = 10.0
ROBOT_WORKSPACE_DEPTH = 3.5

ROOM_X_MIN = -ROOM_WIDTH / 2.0
ROOM_X_MAX = +ROOM_WIDTH / 2.0

# Keep the robot spawn at world origin.
ROBOT_Y_MAX = +ROBOT_WORKSPACE_DEPTH / 2.0   # +1.75
ROBOT_Y_MIN = -ROBOT_WORKSPACE_DEPTH / 2.0   # -1.75

ROOM_Y_MAX = ROBOT_Y_MAX                     # +1.75
ROOM_Y_MIN = ROOM_Y_MAX - ROOM_LENGTH        # -8.25

WALL_HEIGHT = 2.6
WALL_THICKNESS = 0.05

TABLE_HEIGHT = 0.75
TABLE_THICKNESS = 0.05
TABLE_DEPTH = 0.65
TABLE_LEG = 0.055

ROOT = "/World/Lab"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_color(gprim: UsdGeom.Gprim, rgb) -> None:
    gprim.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])


def _box(stage, path, center, size, color, collision=True):
    """Axis-aligned box. `size` is final XYZ dimensions in metres."""
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    cube.AddTranslateOp().Set(Gf.Vec3d(*center))
    cube.AddScaleOp().Set(Gf.Vec3f(*size))
    _set_color(cube, color)

    if collision:
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim())

    return cube


def _cylinder(stage, path, center, radius, height, color, collision=True):
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(height)
    cyl.CreateAxisAttr("Z")
    cyl.AddTranslateOp().Set(Gf.Vec3d(*center))
    _set_color(cyl, color)

    if collision:
        UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())

    return cyl


def _table_piece(stage, root, center_xy, size_xy, color=(0.48, 0.32, 0.20)):
    """One rectangular tabletop + four legs."""
    x, y = center_xy
    sx, sy = size_xy

    _box(
        stage,
        f"{root}/Top",
        (x, y, TABLE_HEIGHT),
        (sx, sy, TABLE_THICKNESS),
        color,
        collision=True,
    )

    leg_height = TABLE_HEIGHT - TABLE_THICKNESS / 2.0
    leg_z = leg_height / 2.0

    margin_x = min(0.08, sx * 0.20)
    margin_y = min(0.08, sy * 0.20)

    dx = sx / 2.0 - margin_x
    dy = sy / 2.0 - margin_y

    for i, (sgn_x, sgn_y) in enumerate(((-1, -1), (-1, 1), (1, -1), (1, 1))):
        _box(
            stage,
            f"{root}/Leg_{i}",
            (x + sgn_x * dx, y + sgn_y * dy, leg_z),
            (TABLE_LEG, TABLE_LEG, leg_height),
            (0.18, 0.18, 0.20),
            collision=True,
        )


# ---------------------------------------------------------------------------
# Room
# ---------------------------------------------------------------------------

def _add_walls(stage):
    cx = 0.0
    cy = (ROOM_Y_MIN + ROOM_Y_MAX) / 2.0

    _box(
        stage, f"{ROOT}/Wall_Left",
        (ROOM_X_MIN, cy, WALL_HEIGHT / 2.0),
        (WALL_THICKNESS, ROOM_LENGTH, WALL_HEIGHT),
        (0.74, 0.76, 0.79),
    )

    _box(
        stage, f"{ROOT}/Wall_Right",
        (ROOM_X_MAX, cy, WALL_HEIGHT / 2.0),
        (WALL_THICKNESS, ROOM_LENGTH, WALL_HEIGHT),
        (0.72, 0.76, 0.72),
    )

    _box(
        stage, f"{ROOT}/Wall_Top",
        (cx, ROOM_Y_MAX, WALL_HEIGHT / 2.0),
        (ROOM_WIDTH, WALL_THICKNESS, WALL_HEIGHT),
        (0.70, 0.72, 0.77),
    )

    _box(
        stage, f"{ROOT}/Wall_Bottom",
        (cx, ROOM_Y_MIN, WALL_HEIGHT / 2.0),
        (ROOM_WIDTH, WALL_THICKNESS, WALL_HEIGHT),
        (0.75, 0.72, 0.69),
    )


# ---------------------------------------------------------------------------
# Robot workspace: intentionally open
# ---------------------------------------------------------------------------

def _add_robot_tables(stage):
    """
    Two L-shaped tables similar to the sketch.
    They hug the side walls so the central robot workspace stays clear.
    """

    # Long wall-side section.
    long_y_size = 2.45
    long_y_center = 0.35

    left_x = ROOM_X_MIN + TABLE_DEPTH / 2.0 + WALL_THICKNESS
    right_x = ROOM_X_MAX - TABLE_DEPTH / 2.0 - WALL_THICKNESS

    _table_piece(
        stage,
        f"{ROOT}/Table_1/Long",
        (left_x, long_y_center),
        (TABLE_DEPTH, long_y_size),
    )

    _table_piece(
        stage,
        f"{ROOT}/Table_2/Long",
        (right_x, long_y_center),
        (TABLE_DEPTH, long_y_size),
    )

    # Short inward-facing L section near the robot/human boundary.
    wing_x_size = 1.10
    wing_y_size = 0.55
    wing_y = ROBOT_Y_MIN + 0.38

    left_wing_x = ROOM_X_MIN + wing_x_size / 2.0 + WALL_THICKNESS
    right_wing_x = ROOM_X_MAX - wing_x_size / 2.0 - WALL_THICKNESS

    _table_piece(
        stage,
        f"{ROOT}/Table_1/Wing",
        (left_wing_x, wing_y),
        (wing_x_size, wing_y_size),
    )

    _table_piece(
        stage,
        f"{ROOT}/Table_2/Wing",
        (right_wing_x, wing_y),
        (wing_x_size, wing_y_size),
    )

    # Plate placeholder on Table 1.
    plate_z = TABLE_HEIGHT + TABLE_THICKNESS / 2.0 + 0.018
    _box(
        stage,
        f"{ROOT}/Table_1/Plate_Placeholder",
        (left_x, 0.65, plate_z),
        (0.28, 0.18, 0.025),
        (0.82, 0.84, 0.87),
        collision=True,
    )

    # Visual drop zone on Table 2.
    drop_z = TABLE_HEIGHT + TABLE_THICKNESS / 2.0 + 0.004
    _box(
        stage,
        f"{ROOT}/Table_2/Drop_Zone",
        (right_x, 0.65, drop_z),
        (0.32, 0.24, 0.006),
        (0.10, 0.45, 0.80),
        collision=False,
    )


def _add_robot_vslam_features(stage):
    """
    Visual-only wall features.
    They add texture/contrast without putting obstacles in the robot workspace.
    """

    # Checker-like panel on the top wall.
    cols = 9
    rows = 4
    tile = 0.17
    start_x = -((cols - 1) * tile) / 2.0
    y = ROOM_Y_MAX - WALL_THICKNESS / 2.0 - 0.012
    z0 = 1.00

    for r in range(rows):
        for c in range(cols):
            x = start_x + c * tile
            z = z0 + r * tile
            color = (0.06, 0.06, 0.06) if (r + c) % 2 == 0 else (0.92, 0.92, 0.92)

            _box(
                stage,
                f"{ROOT}/VSLAM_Top_{r}_{c}",
                (x, y, z),
                (tile * 0.92, 0.012, tile * 0.92),
                color,
                collision=False,
            )

    # Asymmetric colored panels so the two side walls do not look identical.
    x_left = ROOM_X_MIN + WALL_THICKNESS / 2.0 + 0.012
    x_right = ROOM_X_MAX - WALL_THICKNESS / 2.0 - 0.012

    left_panels = [
        (0.95, 1.15, (0.72, 0.12, 0.12)),
        (0.10, 1.55, (0.12, 0.48, 0.18)),
        (-0.75, 1.20, (0.12, 0.25, 0.74)),
    ]

    right_panels = [
        (0.75, 1.55, (0.80, 0.55, 0.08)),
        (-0.25, 1.10, (0.46, 0.18, 0.68)),
    ]

    for i, (y0, z0, color) in enumerate(left_panels):
        _box(
            stage,
            f"{ROOT}/VSLAM_Left_{i}",
            (x_left, y0, z0),
            (0.012, 0.42, 0.34),
            color,
            collision=False,
        )

    for i, (y0, z0, color) in enumerate(right_panels):
        _box(
            stage,
            f"{ROOT}/VSLAM_Right_{i}",
            (x_right, y0, z0),
            (0.012, 0.48, 0.32),
            color,
            collision=False,
        )


# ---------------------------------------------------------------------------
# Human workspace: static office-like clutter
# ---------------------------------------------------------------------------

def _add_human_workspace(stage):
    """
    No moving human yet.
    Static office-like furniture creates realistic VSLAM visual clutter.
    The center remains navigable so later a human actor can traverse it.
    """

    # Side desks.
    _box(
        stage,
        f"{ROOT}/Office/Desk_Left",
        (-1.65, -3.00, 0.38),
        (0.70, 1.45, 0.76),
        (0.46, 0.33, 0.21),
        collision=True,
    )

    _box(
        stage,
        f"{ROOT}/Office/Desk_Right",
        (1.65, -3.15, 0.38),
        (0.70, 1.35, 0.76),
        (0.46, 0.33, 0.21),
        collision=True,
    )

    # Shelves near the lower corners.
    _box(
        stage,
        f"{ROOT}/Office/Shelf_Left",
        (-2.05, -6.25, 0.95),
        (0.38, 1.20, 1.90),
        (0.55, 0.54, 0.50),
        collision=True,
    )

    _box(
        stage,
        f"{ROOT}/Office/Shelf_Right",
        (2.05, -6.00, 0.95),
        (0.38, 1.25, 1.90),
        (0.53, 0.53, 0.49),
        collision=True,
    )

    # Cabinet and printer stand along the bottom.
    _box(
        stage,
        f"{ROOT}/Office/Cabinet",
        (-0.65, -7.70, 0.52),
        (0.95, 0.38, 1.04),
        (0.67, 0.69, 0.72),
        collision=True,
    )

    _box(
        stage,
        f"{ROOT}/Office/Printer_Stand",
        (0.75, -7.65, 0.38),
        (0.60, 0.42, 0.76),
        (0.60, 0.62, 0.66),
        collision=True,
    )

    # A couple of boxes.
    _box(
        stage,
        f"{ROOT}/Office/Box_A",
        (-0.75, -4.75, 0.17),
        (0.34, 0.30, 0.34),
        (0.64, 0.27, 0.17),
        collision=True,
    )

    _box(
        stage,
        f"{ROOT}/Office/Box_B",
        (0.80, -5.00, 0.15),
        (0.30, 0.28, 0.30),
        (0.15, 0.38, 0.66),
        collision=True,
    )

    # Simple office chairs.
    chairs = [
        (-1.15, -2.35),
        (1.15, -2.45),
        (-0.20, -5.65),
    ]

    for i, (x, y) in enumerate(chairs):
        _cylinder(
            stage,
            f"{ROOT}/Office/Chair_{i}/Base",
            (x, y, 0.22),
            radius=0.16,
            height=0.44,
            color=(0.18, 0.18, 0.21),
            collision=True,
        )

        _box(
            stage,
            f"{ROOT}/Office/Chair_{i}/Seat",
            (x, y, 0.48),
            (0.34, 0.34, 0.08),
            (0.15, 0.15, 0.18),
            collision=True,
        )

    # Visual-only wall posters in the office area.
    x = ROOM_X_MAX - WALL_THICKNESS / 2.0 - 0.012
    posters = [
        (-3.1, 1.20, (0.70, 0.14, 0.14)),
        (-4.35, 1.55, (0.12, 0.52, 0.20)),
        (-5.55, 1.28, (0.15, 0.28, 0.76)),
        (-6.85, 1.62, (0.86, 0.60, 0.08)),
    ]

    for i, (y, z, color) in enumerate(posters):
        _box(
            stage,
            f"{ROOT}/Office/Poster_{i}",
            (x, y, z),
            (0.012, 0.55, 0.40),
            color,
            collision=False,
        )


# ---------------------------------------------------------------------------
# Optional visual boundary
# ---------------------------------------------------------------------------

def _add_workspace_boundary(stage):
    # Thin visual-only stripe at Y = -1.75, matching the blue dotted line.
    _box(
        stage,
        f"{ROOT}/Robot_Human_Boundary",
        (0.0, ROBOT_Y_MIN, 0.002),
        (ROOM_WIDTH - 0.15, 0.025, 0.004),
        (0.10, 0.48, 0.90),
        collision=False,
    )


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def add_lab_environment(stage) -> None:
    UsdGeom.Xform.Define(stage, ROOT)

    _add_walls(stage)
    _add_workspace_boundary(stage)

    _add_robot_tables(stage)
    add_pick_plate(stage)
    _add_robot_vslam_features(stage)

    _add_human_workspace(stage)

    dome = UsdLux.DomeLight.Define(stage, f"{ROOT}/DomeLight")
    dome.CreateIntensityAttr(480.0)
    dome.CreateColorAttr(Gf.Vec3f(0.95, 0.96, 1.0))

    print(
        "[LabEnvironment] 5.0m x 10.0m lab loaded | "
        "robot workspace=3.5m | human workspace=6.5m | "
        "moving human actor=disabled"
    )
