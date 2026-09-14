# SPDX-License-Identifier: Apache-2.0
"""
Fixed dynamic AprilTag plate for RBY1 Isaac Sim.

IMPORTANT FIX:
- /World/Lab/PickPlate is now an Xform + RigidBody only.
- The scaled Cube body is a child.
- AprilTag mesh uses LOCAL coordinates and no longer inherits the body's scale.
"""

from __future__ import annotations

from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

PLATE_LENGTH = 0.180
PLATE_WIDTH = 0.060
PLATE_THICKNESS = 0.025
PLATE_MASS_KG = 0.250

APRILTAG_SIZE_M = 0.045
TAG_BOARD_SIZE_M = 0.055
TAG_ID = 10
TAG_FAMILY = "tag36h11"

# AprilTag centered on the plate.
TAG_X_OFFSET = 0.0

DEFAULT_X = -2.125
DEFAULT_Y = 0.650

TABLE_HEIGHT = 0.750
TABLE_THICKNESS = 0.050
INITIAL_GAP = 0.002
DEFAULT_Z = (
    TABLE_HEIGHT
    + TABLE_THICKNESS / 2.0
    + PLATE_THICKNESS / 2.0
    + INITIAL_GAP
)

PLATE_PRIM_PATH = "/World/Lab/PickPlate"
OLD_PLACEHOLDER_PATH = "/World/Lab/Table_1/Plate_Placeholder"

DEFAULT_TAG_TEXTURE = "/opt/rby1-sim-isaac/assets/tags/tag36h11_id10.png"


def _set_display_color(gprim: UsdGeom.Gprim, rgb):
    gprim.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])


def _create_tag_material(stage, material_path: str, texture_path: str):
    material = UsdShade.Material.Define(stage, material_path)

    surface = UsdShade.Shader.Define(stage, f"{material_path}/PreviewSurface")
    surface.CreateIdAttr("UsdPreviewSurface")
    surface.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.7)
    surface.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    st_reader = UsdShade.Shader.Define(stage, f"{material_path}/STReader")
    st_reader.CreateIdAttr("UsdPrimvarReader_float2")
    st_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")

    texture = UsdShade.Shader.Define(stage, f"{material_path}/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(texture_path))
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
        st_reader.ConnectableAPI(), "result"
    )
    texture.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
    texture.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")

    surface.CreateInput(
        "diffuseColor", Sdf.ValueTypeNames.Color3f
    ).ConnectToSource(texture.ConnectableAPI(), "rgb")

    material.CreateSurfaceOutput().ConnectToSource(
        surface.ConnectableAPI(), "surface"
    )
    return material


def _create_tag_quad(stage, path: str, local_x: float, local_y: float, local_z: float,
                     texture_path: str):
    half = TAG_BOARD_SIZE_M / 2.0

    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([
        Gf.Vec3f(local_x - half, local_y - half, local_z),
        Gf.Vec3f(local_x + half, local_y - half, local_z),
        Gf.Vec3f(local_x + half, local_y + half, local_z),
        Gf.Vec3f(local_x - half, local_y + half, local_z),
    ])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)

    # Explicit upward normal.
    mesh.CreateNormalsAttr([Gf.Vec3f(0, 0, 1)] * 4)
    mesh.SetNormalsInterpolation(UsdGeom.Tokens.vertex)

    primvars = UsdGeom.PrimvarsAPI(mesh)
    st = primvars.CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.vertex,
    )
    st.Set([
        Gf.Vec2f(0.0, 0.0),
        Gf.Vec2f(1.0, 0.0),
        Gf.Vec2f(1.0, 1.0),
        Gf.Vec2f(0.0, 1.0),
    ])

    material = _create_tag_material(
        stage,
        f"{path}_Material",
        texture_path,
    )
    UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)
    return mesh


def add_pick_plate(
    stage,
    x: float = DEFAULT_X,
    y: float = DEFAULT_Y,
    z: float = DEFAULT_Z,
    texture_path: str = DEFAULT_TAG_TEXTURE,
):
    # Remove old static placeholder.
    old = stage.GetPrimAtPath(OLD_PLACEHOLDER_PATH)
    if old.IsValid():
        stage.RemovePrim(OLD_PLACEHOLDER_PATH)
        print(f"[PickPlate] Removed old placeholder: {OLD_PLACEHOLDER_PATH}")

    # Idempotent recreation.
    old_plate = stage.GetPrimAtPath(PLATE_PRIM_PATH)
    if old_plate.IsValid():
        stage.RemovePrim(PLATE_PRIM_PATH)

    # Root: transform + rigid body, NO scale here.
    root = UsdGeom.Xform.Define(stage, PLATE_PRIM_PATH)
    root.AddTranslateOp().Set(Gf.Vec3d(x, y, z))

    root_prim = root.GetPrim()
    rigid = UsdPhysics.RigidBodyAPI.Apply(root_prim)
    rigid.CreateRigidBodyEnabledAttr(True)

    mass = UsdPhysics.MassAPI.Apply(root_prim)
    mass.CreateMassAttr(PLATE_MASS_KG)

    # Body: local geometry only.
    body = UsdGeom.Cube.Define(stage, f"{PLATE_PRIM_PATH}/Body")
    body.CreateSizeAttr(1.0)
    body.AddScaleOp().Set(
        Gf.Vec3f(PLATE_LENGTH, PLATE_WIDTH, PLATE_THICKNESS)
    )
    _set_display_color(body, (0.78, 0.80, 0.83))
    UsdPhysics.CollisionAPI.Apply(body.GetPrim())

    # Centered AprilTag: local coordinates relative to rigid-body root.
    tag_z = PLATE_THICKNESS / 2.0 + 0.0008
    _create_tag_quad(
        stage,
        f"{PLATE_PRIM_PATH}/AprilTag_{TAG_ID}",
        TAG_X_OFFSET,
        0.0,
        tag_z,
        texture_path,
    )

    print(
        "[PickPlate] FIXED plate loaded | "
        f"{PLATE_LENGTH*1000:.0f}x{PLATE_WIDTH*1000:.0f}x{PLATE_THICKNESS*1000:.0f} mm | "
        f"{TAG_FAMILY} ID={TAG_ID} | black tag={APRILTAG_SIZE_M*1000:.0f} mm"
    )

    return {
        "plate_path": PLATE_PRIM_PATH,
        "body_path": f"{PLATE_PRIM_PATH}/Body",
        "tag_path": f"{PLATE_PRIM_PATH}/AprilTag_{TAG_ID}",
        "tag_id": TAG_ID,
        "tag_family": TAG_FAMILY,
        "tag_size_m": APRILTAG_SIZE_M,
        "recommended_grasp_offset_plate": (0.0, 0.0, 0.0),
    }
