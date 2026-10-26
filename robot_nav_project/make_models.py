"""make_models.py - build the realistic 3D models in Blender and export them
for the simulator.

HOW TO USE (in Blender, takes about a minute):
  1. Open Blender, click the "Scripting" tab at the top.
  2. In the text editor click "Open", pick this file (make_models.py) from
     the project folder, then click "Run Script" (the play button).
  3. When it is done, the project folder has a "models" folder full of .obj
     files, and Blender shows all the models laid out so you can look at them.
  4. Run the simulator as usual (config.USE_BLENDER_MODELS = True).

What Blender adds: every part is smoothed (people, trees) or given rounded
edges (cars, benches, shelters) before it is exported. The shapes themselves
come from model_specs.py, so they always fit the simulator's sizes.

Changed a model by hand in Blender and want to keep your edits? Set
REBUILD = False below and run the script again: it then only re-exports the
objects that are already in the "robot_nav_models" collection.

No Blender? `python make_models.py` writes the same models without the
smoothing (still much nicer than plain boxes).
"""
import os
import sys

REBUILD = True                    # False = only re-export what is already in Blender (keeps hand edits)
COLLECTION = "robot_nav_models"

# preview colours inside Blender only (the simulator picks its own colours)
PREVIEW = {
    "pants": (0.15, 0.2, 0.4), "shoe": (0.08, 0.08, 0.08), "arm": (0.75, 0.2, 0.2), "hand": (0.9, 0.7, 0.55),
    "pelvis": (0.15, 0.2, 0.4), "torso": (0.75, 0.2, 0.2), "neck": (0.9, 0.7, 0.55), "head": (0.9, 0.7, 0.55),
    "hair": (0.2, 0.12, 0.06), "cap": (0.1, 0.1, 0.3), "coat": (0.35, 0.28, 0.2), "backpack": (0.2, 0.3, 0.6),
    "bag": (0.35, 0.2, 0.1), "umbrella": (0.1, 0.1, 0.12), "body": (0.7, 0.1, 0.1), "glass": (0.15, 0.2, 0.27),
    "trim": (0.08, 0.08, 0.08), "tyres": (0.04, 0.04, 0.04), "rims": (0.75, 0.75, 0.78),
    "headlights": (1.0, 1.0, 0.7), "taillights": (0.8, 0.1, 0.1), "bark": (0.4, 0.28, 0.18), "leaf": (0.2, 0.55, 0.2),
    "metal": (0.25, 0.25, 0.28), "light": (1.0, 0.95, 0.6), "wood": (0.55, 0.38, 0.22), "red": (0.85, 0.1, 0.1),
    "ad": (0.9, 0.9, 0.95), "stone": (0.7, 0.7, 0.68), "water": (0.35, 0.6, 0.85), "concrete": (0.7, 0.7, 0.68),
    "pole": (0.25, 0.25, 0.28), "housing": (0.12, 0.12, 0.12), "yellow": (0.9, 0.7, 0.1), "green": (0.1, 0.9, 0.3),
    "box": (0.9, 0.1, 0.1), "blue": (0.1, 0.3, 0.7), "white": (0.95, 0.95, 0.95),
}


def project_dir():
    """The folder this script is in (works from Blender's text editor too)."""
    try:
        import bpy
        for text in bpy.data.texts:
            if text.filepath and os.path.basename(text.filepath) == "make_models.py":
                return os.path.dirname(bpy.path.abspath(text.filepath))
    except ImportError:
        pass
    if "__file__" in globals() and os.path.exists(__file__):
        return os.path.dirname(os.path.abspath(__file__))
    return os.getcwd()


PROJECT = project_dir()
OUT = os.path.join(PROJECT, "models")
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)
import model_specs  # noqa: E402  (needs the project folder on the path first)

if "model_specs" in sys.modules:           # pick up edits when re-running inside Blender
    import importlib
    importlib.reload(model_specs)


def preview_colour(name):
    for key, col in PREVIEW.items():
        if name.endswith(key) or f"_{key}" in name:
            return col
    return (0.6, 0.6, 0.6)


# ======================================================================= without Blender
def export_plain():
    meshes, finish = model_specs.all_models()
    os.makedirs(OUT, exist_ok=True)
    for name, mesh in meshes.items():
        model_specs.write_obj(os.path.join(OUT, name + ".obj"), mesh, smooth=finish[name] == "smooth")
    print(f"Wrote {len(meshes)} models to {OUT} (no Blender: shapes without extra smoothing)")


# ======================================================================= inside Blender
def export_blender():
    import bpy

    os.makedirs(OUT, exist_ok=True)
    scene = bpy.context.scene
    col = bpy.data.collections.get(COLLECTION)

    if REBUILD:
        if col is not None:                        # throw away the old models
            for obj in list(col.objects):
                bpy.data.objects.remove(obj, do_unlink=True)
        else:
            col = bpy.data.collections.new(COLLECTION)
            scene.collection.children.link(col)
        meshes, finish = model_specs.all_models()
        objects = []
        for name, m in meshes.items():
            data = bpy.data.meshes.new(name)
            data.from_pydata([tuple(p) for p in m.v], [], [tuple(f) for f in m.f])
            data.update()
            obj = bpy.data.objects.new(name, data)
            col.objects.link(obj)
            _finish(obj, finish[name])
            _material(obj, name)
            objects.append(obj)
    else:
        if col is None:
            print("No 'robot_nav_models' collection found - set REBUILD = True")
            return
        objects = list(col.objects)

    exported = 0
    for obj in objects:
        keep = obj.location.copy()
        obj.location = (0.0, 0.0, 0.0)              # export each part in its own coordinates
        _export_one(bpy, obj, os.path.join(OUT, obj.name + ".obj"))
        obj.location = keep
        exported += 1
    if REBUILD:
        _lay_out(objects)
    try:
        bpy.ops.wm.save_as_mainfile(filepath=os.path.join(OUT, "robot_nav_models.blend"), copy=True)
    except Exception as err:                       # saving the .blend is only a convenience
        print("Could not save robot_nav_models.blend:", err)
    print(f"Exported {exported} models to {OUT}")


def _finish(obj, kind):
    """Smooth organic parts, round off the edges of hard ones."""
    data = obj.data
    if kind == "smooth":
        data.polygons.foreach_set("use_smooth", [True] * len(data.polygons))
        mod = obj.modifiers.new("Smooth", "SUBSURF")
        mod.levels = 1
        mod.render_levels = 1
    elif kind == "bevel":
        xs = [v.co.x for v in data.vertices]
        ys = [v.co.y for v in data.vertices]
        zs = [v.co.z for v in data.vertices]
        smallest = min(max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))
        mod = obj.modifiers.new("Round edges", "BEVEL")
        mod.width = max(0.004, min(0.05, smallest * 0.12))
        mod.segments = 2
        mod.limit_method = "ANGLE"


def _material(obj, name):
    mat = bpy_material(name)
    if mat is not None:
        obj.data.materials.append(mat)


def bpy_material(name):
    import bpy
    rgb = preview_colour(name)
    key = "rn_" + "_".join(f"{c:.2f}" for c in rgb)
    mat = bpy.data.materials.get(key) or bpy.data.materials.new(key)
    mat.diffuse_color = (rgb[0], rgb[1], rgb[2], 1.0)
    return mat


def _export_one(bpy, obj, path):
    for o in bpy.context.view_layer.objects:
        o.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    if hasattr(bpy.ops.wm, "obj_export"):          # Blender 3.2 and newer
        bpy.ops.wm.obj_export(filepath=path, export_selected_objects=True, forward_axis="Y", up_axis="Z",
                              apply_modifiers=True, export_materials=False, export_normals=True,
                              export_uv=False, export_triangulated_mesh=True)
    else:                                          # older Blender
        bpy.ops.export_scene.obj(filepath=path, use_selection=True, axis_forward="Y", axis_up="Z",
                                 use_mesh_modifiers=True, use_materials=False, use_normals=True,
                                 use_triangles=True)
    obj.select_set(False)


def _lay_out(objects):
    """Put the models side by side in Blender so you can look at them:
    people assembled in a row, then cars, then street objects."""
    by_name = {o.name: o for o in objects}
    for name, obj in by_name.items():
        obj.location = (0.0, -30.0, 0.0)           # anything not placed below goes out of the way
    # a person, put together
    seen = set()
    for part, mesh, _, (dx, dy, z), _ in model_specs.PERSON_LAYOUT:
        if mesh in seen:
            continue                               # each mesh exists once; show the left side
        seen.add(mesh)
        by_name[mesh].location = (dx, dy, z)
    x = 1.0
    for extra, (mesh, _, (dx, dy, z)) in model_specs.PERSON_EXTRAS.items():
        by_name[mesh].location = (dx, dy + x, z)
        x += 1.0
    # vehicles in a row
    y = 8.0
    for kind in model_specs.VEHICLE_SIZES:
        for part in model_specs.VEHICLE_PARTS:
            by_name[f"car_{kind}_{part}"].location = (0.0, y, 0.0)
        y += model_specs.VEHICLE_SIZES[kind][1] + 2.5
    # street objects in a row
    x = 8.0
    for model, parts in model_specs.STREET_PARTS.items():
        for part in parts:
            by_name[f"{model}_{part}"].location = (x, 0.0, 0.0)
        x += 5.0 if model != "fountain" else 6.0


if __name__ == "__main__":
    try:
        import bpy  # noqa: F401
        in_blender = True
    except ImportError:
        in_blender = False
    if in_blender:
        export_blender()
    else:
        export_plain()
