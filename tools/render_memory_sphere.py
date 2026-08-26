from pathlib import Path
import math
import random

import bpy
from mathutils import Vector


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "assets" / "memory-sphere"
OUTPUT.mkdir(parents=True, exist_ok=True)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        for datablock in list(datablocks):
            if datablock.users == 0:
                datablocks.remove(datablock)


def look_at(obj, target=(0.0, 0.0, 0.0)):
    direction = Vector(target) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def make_material(name, color, roughness=0.7, metallic=0.0):
    material = bpy.data.materials.new(name)
    material.use_nodes = True
    shader = material.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = color
    shader.inputs["Roughness"].default_value = roughness
    shader.inputs["Metallic"].default_value = metallic
    return material


def make_sphere():
    bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=6, radius=1.35, location=(0.0, 0.0, 0.0))
    sphere = bpy.context.active_object
    sphere.name = "Memory Sphere"

    for vertex in sphere.data.vertices:
        normal = vertex.co.normalized()
        wave = (
            0.012 * math.sin(normal.x * 3.1 + normal.z * 2.2)
            + 0.008 * math.sin(normal.y * 4.2 - normal.x * 1.4)
            + 0.006 * math.cos(normal.z * 3.7 + normal.y * 1.1)
        )
        dent_a = -0.026 * math.exp(
            -((normal.x - 0.34) ** 2 + (normal.y + 0.28) ** 2 + (normal.z - 0.46) ** 2) / 0.2
        )
        dent_b = -0.014 * math.exp(
            -((normal.x + 0.55) ** 2 + (normal.y - 0.16) ** 2 + (normal.z + 0.12) ** 2) / 0.24
        )
        vertex.co *= 1.0 + wave + dent_a + dent_b

    sphere.scale = (1.025, 0.995, 0.975)
    for polygon in sphere.data.polygons:
        polygon.use_smooth = True

    sphere.data.materials.append(make_material("Sphere Surface", (0.006, 0.009, 0.015, 1.0), 0.76, 0.04))
    return sphere


def make_particle_cloud():
    random.seed(29)
    vertices = []
    faces = []

    for index in range(150):
        progress = random.random() ** 1.65
        theta = random.uniform(-1.05, 1.05)
        radius = 1.28 + progress * 1.15 + random.uniform(-0.08, 0.08)
        x = radius * math.cos(theta) + progress * 0.34
        y = random.uniform(-0.42, 0.42) * (0.45 + progress)
        z = radius * math.sin(theta) * 0.62 + random.uniform(-0.1, 0.1)
        size = random.uniform(0.008, 0.023) * (1.0 - progress * 0.45)
        base = len(vertices)
        vertices.extend(
            [
                (x + size, y, z - size),
                (x - size, y, z - size),
                (x, y + size, z + size),
                (x, y - size, z + size),
            ]
        )
        faces.extend(
            [
                (base, base + 1, base + 2),
                (base, base + 3, base + 1),
                (base, base + 2, base + 3),
                (base + 1, base + 3, base + 2),
            ]
        )

    mesh = bpy.data.meshes.new("Particle Wake Mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    cloud = bpy.data.objects.new("Particle Wake", mesh)
    bpy.context.collection.objects.link(cloud)
    cloud.data.materials.append(make_material("Particle Material", (0.12, 0.2, 0.42, 1.0), 0.48, 0.0))
    return cloud


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
    scene.render.film_transparent = True
    scene.render.image_settings.color_depth = "8"
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = 1
    scene.render.pixel_aspect_y = 1
    scene.render.fps = 18
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"

    world = bpy.data.worlds.new("Memory Sphere World") if not bpy.data.worlds else bpy.data.worlds[0]
    scene.world = world
    world.use_nodes = True
    background = world.node_tree.nodes.get("Background")
    background.inputs["Color"].default_value = (0.005, 0.008, 0.014, 1.0)
    background.inputs["Strength"].default_value = 0.01

    camera_data = bpy.data.cameras.new("Memory Sphere Camera")
    camera = bpy.data.objects.new("Memory Sphere Camera", camera_data)
    bpy.context.collection.objects.link(camera)
    camera.location = (0.0, -6.4, 0.12)
    camera_data.type = "ORTHO"
    camera_data.ortho_scale = 3.75
    look_at(camera, (0.0, 0.0, 0.02))
    scene.camera = camera

    add_area_light("Cool Key", (-3.7, -4.2, 3.5), (0.63, 0.78, 1.0), 980, 4.2)
    add_area_light("Blue Rim", (3.5, 1.0, 1.8), (0.14, 0.32, 1.0), 820, 3.0)
    add_area_light("Soft Fill", (0.0, -2.0, -4.2), (0.32, 0.4, 0.55), 120, 5.0)
    return scene


def render_stills(scene, sphere):
    rotations = [
        (math.radians(7), math.radians(-28), math.radians(-5)),
        (math.radians(-12), math.radians(38), math.radians(11)),
        (math.radians(18), math.radians(112), math.radians(-13)),
    ]

    for index, rotation in enumerate(rotations, start=1):
        sphere.rotation_euler = rotation
        scene.render.filepath = str(OUTPUT / f"still-{index:02d}.png")
        bpy.ops.render.render(write_still=True)


clear_scene()
scene = configure_scene()
sphere = make_sphere()
make_particle_cloud()
render_stills(scene, sphere)
bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT / "memory-sphere-blockout.blend"))
print(f"Rendered memory sphere blockouts to {OUTPUT}")
