from pathlib import Path
import math
import struct
import zlib


ROOT = Path(__file__).resolve().parent.parent
SOURCE = Path("/private/tmp/kda-memory-field-source")
OUTPUT = Path("/private/tmp/kda-memory-field-ascii")
OUTPUT.mkdir(parents=True, exist_ok=True)
FRAME_COUNT = 48

GLYPHS = {
    "0": (
        "0011100",
        "0100010",
        "1000001",
        "1000001",
        "1000001",
        "1000001",
        "1000001",
        "1000001",
        "1000001",
        "0100010",
        "0011100",
    ),
    "1": (
        "0001000",
        "0011000",
        "0101000",
        "0001000",
        "0001000",
        "0001000",
        "0001000",
        "0001000",
        "0001000",
        "0001000",
        "0111110",
    ),
}


def paeth(a, b, c):
    p = a + b - c
    pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
    return a if pa <= pb and pa <= pc else b if pb <= pc else c


def read_rgba_png(path):
    data = path.read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Not a PNG: {path}")
    offset = 8
    compressed = bytearray()
    width = height = None
    while offset < len(data):
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        kind = data[offset + 4:offset + 8]
        payload = data[offset + 8:offset + 8 + length]
        offset += length + 12
        if kind == b"IHDR":
            width, height, depth, color, _, _, interlace = struct.unpack(">IIBBBBB", payload)
            if (depth, color, interlace) != (8, 6, 0):
                raise ValueError(f"Expected non-interlaced 8-bit RGBA PNG, got {(depth, color, interlace)}")
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"IEND":
            break

    raw = zlib.decompress(bytes(compressed))
    stride = width * 4
    rows = []
    cursor = 0
    previous = bytearray(stride)
    for _ in range(height):
        filter_type = raw[cursor]
        cursor += 1
        encoded = raw[cursor:cursor + stride]
        cursor += stride
        row = bytearray(stride)
        for index, value in enumerate(encoded):
            left = row[index - 4] if index >= 4 else 0
            up = previous[index]
            upper_left = previous[index - 4] if index >= 4 else 0
            if filter_type == 0:
                decoded = value
            elif filter_type == 1:
                decoded = value + left
            elif filter_type == 2:
                decoded = value + up
            elif filter_type == 3:
                decoded = value + ((left + up) // 2)
            elif filter_type == 4:
                decoded = value + paeth(left, up, upper_left)
            else:
                raise ValueError(f"Unsupported PNG filter {filter_type}")
            row[index] = decoded & 255
        rows.append(row)
        previous = row
    return width, height, rows


def png_chunk(kind, payload):
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)


def write_rgba_png(path, width, height, pixels):
    raw = bytearray()
    stride = width * 4
    for y in range(height):
        raw.append(0)
        raw.extend(pixels[y * stride:(y + 1) * stride])
    data = b"\x89PNG\r\n\x1a\n"
    data += png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
    data += png_chunk(b"IDAT", zlib.compress(bytes(raw), 9))
    data += png_chunk(b"IEND", b"")
    path.write_bytes(data)


def stable_noise(x, y, seed=0):
    value = math.sin(x * 12.9898 + y * 78.233 + seed * 37.719) * 43758.5453
    return value - math.floor(value)


def draw_glyph(pixels, width, height, char, x, y, alpha, scale=2):
    pattern = GLYPHS[char]
    glyph_width = len(pattern[0]) * scale
    glyph_height = len(pattern) * scale
    left = int(x - glyph_width / 2)
    top = int(y - glyph_height / 2)
    for gy, row in enumerate(pattern):
        for gx, enabled in enumerate(row):
            if enabled != "1":
                continue
            for sy in range(scale):
                for sx in range(scale):
                    px = left + gx * scale + sx
                    py = top + gy * scale + sy
                    if 0 <= px < width and 0 <= py < height:
                        index = (py * width + px) * 4
                        pixels[index:index + 4] = bytes((17, 23, 19, max(pixels[index + 3], alpha)))


def convert_frame(frame):
    width, height, rows = read_rgba_png(SOURCE / f"frame-{frame:03d}.png")
    output = bytearray(width * height * 4)
    progress = frame / FRAME_COUNT
    phase = math.tau * progress
    step = 17
    center_x = width * 0.5
    center_y = height * 0.5
    sphere_radius = min(width, height) * 0.365

    for x in range(step // 2, width, step):
        column = x // step
        normalized_x = (x - center_x) / sphere_radius
        edge_column = abs(normalized_x) > 0.6
        if stable_noise(column, 2, 67) < (0.06 if edge_column else 0.14):
            continue

        period = 10 + int(stable_noise(column, 5, 71) * 5)
        trail_length = 5 + int(stable_noise(column, 7, 73) * 5)
        fall_offset = progress * step * period
        phase_offset = int(stable_noise(column, 11, 79) * period)

        for base_row in range(-period - 2, height // step + 2):
            y = int(base_row * step + fall_offset)
            if y < 6 or y >= height - 6:
                continue

            stream_position = (base_row + phase_offset) % period
            if stream_position >= trail_length:
                continue

            samples = []
            for oy in (-4, 0, 4):
                for ox in (-4, 0, 4):
                    index = (x + ox) * 4
                    pixel = rows[y + oy][index:index + 4]
                    samples.append(pixel)
            source_alpha = sum(pixel[3] for pixel in samples) / (255 * len(samples))
            if source_alpha < 0.08:
                continue
            luminance = sum((pixel[0] * 0.22 + pixel[1] * 0.7 + pixel[2] * 0.08) for pixel in samples) / (255 * len(samples))

            normalized_y = (y - center_y) / sphere_radius
            radial = min(1, math.sqrt(normalized_x * normalized_x + normalized_y * normalized_y))
            center_weight = max(0, 1 - radial * radial)
            rim = max(0, min(1, (radial - 0.72) / 0.25))
            right_crescent = rim * max(0, min(1, (normalized_x + 0.08) / 0.92))
            light_direction_x = -0.72 + math.sin(phase) * 0.08
            light_direction_y = -0.2 + math.cos(phase) * 0.06
            terminator = max(0, min(1, 0.58 - normalized_x * light_direction_x - normalized_y * light_direction_y))

            trail_progress = stream_position / max(1, trail_length - 1)
            trail_alpha = 0.28 + math.pow(trail_progress, 1.3) * 0.72
            light_alpha = 70 + min(1, luminance * 2.05) * 158
            volume_alpha = 24 * center_weight + 72 * right_crescent
            alpha = int((light_alpha + volume_alpha) * trail_alpha * (0.72 + terminator * 0.28))
            alpha = min(245, alpha)
            char = "1" if stable_noise(column, stream_position, 83) > 0.5 else "0"
            draw_glyph(output, width, height, char, x, y, alpha, 1)

    destination = OUTPUT / f"frame-{frame:03d}.png"
    write_rgba_png(destination, width, height, output)
    if frame == 0:
        write_rgba_png(ROOT / "assets" / "memory-sphere" / "memory-field-poster.png", width, height, output)


for frame_index in range(FRAME_COUNT):
    convert_frame(frame_index)
print(f"Converted {FRAME_COUNT} ASCII frames to {OUTPUT}")
