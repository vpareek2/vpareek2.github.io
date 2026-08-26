from pathlib import Path
import math

import bpy
from mathutils import Vector


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = Path("/private/tmp/kda-memory-field-source")
OUTPUT.mkdir(parents=True, exist_ok=True)
FRAME_COUNT = 48


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def look_at(obj, target=(0.0, 0.0, 0.0)):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def make_sphere():
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=6, radius=1.34)
    sphere = bpy.context.active_object
    sphere.name = "Memory Field"

    # Only broad asymmetry. The silhouette must stay quiet once rasterized.
    for vertex in sphere.data.vertices:
        normal = vertex.co.normalized()
        drift = (
            0.009 * math.sin(normal.x * 2.2 + normal.z * 1.7)
            + 0.006 * math.cos(normal.y * 2.8 - normal.x * 1.2)
        )
        shallow_dent = -0.018 * math.exp(
            -((normal.x - 0.48) ** 2 + (normal.y + 0.18) ** 2 + (normal.z - 0.3) ** 2) / 0.28
        )
        vertex.co *= 1 + drift + shallow_dent

    sphere.scale = (1.035, 0.985, 0.955)
    for polygon in sphere.data.polygons:
        polygon.use_smooth = True

    material = bpy.data.materials.new("Memory Surface")
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links
    shader = nodes.get("Principled BSDF")
    shader.inputs["Roughness"].default_value = 0.72
    shader.inputs["Metallic"].default_value = 0.02

    noise = nodes.new("ShaderNodeTexNoise")
    noise.inputs["Scale"].default_value = 1.55
    noise.inputs["Detail"].default_value = 2.1
    noise.inputs["Roughness"].default_value = 0.52
    ramp = nodes.new("ShaderNodeValToRGB")
    ramp.color_ramp.elements[0].position = 0.22
    ramp.color_ramp.elements[0].color = (0.015, 0.022, 0.035, 1)
    ramp.color_ramp.elements[1].position = 0.82
    ramp.color_ramp.elements[1].color = (0.19, 0.24, 0.34, 1)
    links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
    links.new(ramp.outputs["Color"], shader.inputs["Base Color"])
    sphere.data.materials.append(material)
    return sphere


def add_area_light(name, location, color, energy, size):
    data = bpy.data.lights.new(name, type="AREA")
    data.color = color
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    light = bpy.data.objects.new(name, data)
    bpy.context.collection.objects.link(light)
    light.location = location
    look_at(light)
    return light


def configure_scene():
    scene = bpy.context.scene
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 600
    scene.render.resolution_y = 500
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = True
    scene.render.fps = 16
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"

    world = bpy.data.worlds.new("Memory Field World") if not bpy.data.worlds else bpy.data.worlds[0]
    scene.world = world
    world.use_nodes = True
    world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.006

    camera_data = bpy.data.cameras.new("Memory Field Camera")
    camera = bpy.data.objects.new("Memory Field Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = (0, -6.4, 0.03)
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 3.72
    look_at(camera)
    scene.camera = camera

    key = add_area_light("Key", (-3.8, -4.4, 3.2), (0.76, 0.84, 1.0), 1050, 4.0)
    rim = add_area_light("Rim", (3.4, 0.5, 1.6), (0.25, 0.42, 1.0), 760, 2.8)
    add_area_light("Fill", (0.0, -2.0, -4.0), (0.4, 0.48, 0.62), 105, 5.0)
    return scene, key, rim


def render_loop(scene, sphere, key, rim):
    for frame in range(FRAME_COUNT):
        phase = math.tau * frame / FRAME_COUNT
        sphere.rotation_euler = (
            math.radians(-7) + math.sin(phase) * math.radians(4),
            phase,
            math.radians(5) + math.cos(phase) * math.radians(3),
        )
        key.location = (-3.8 + math.sin(phase) * 0.85, -4.4, 3.2 + math.cos(phase) * 0.45)
        rim.location = (3.4, 0.5 + math.sin(phase) * 0.5, 1.6 + math.cos(phase) * 0.35)
        look_at(key)
        look_at(rim)
        scene.render.filepath = str(OUTPUT / f"frame-{frame:03d}.png")
        bpy.ops.render.render(write_still=True)


clear_scene()
scene, key, rim = configure_scene()
sphere = make_sphere()
render_loop(scene, sphere, key, rim)
print(f"Rendered {FRAME_COUNT} source frames to {OUTPUT}")
