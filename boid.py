"""Interactive flocking simulation with synchronous, fixed-step updates."""

import argparse
from collections import defaultdict
import math

import numpy as np
import pygame

WIDTH, HEIGHT = 1080, 720
COLORS = ('#FF5555', '#55FF55', '#5599FF')


class Boid:
    def __init__(self, size, position=None, max_speed=100.0, color=None,
                 rng=None, index=0):
        self.size = size
        self.position = np.zeros(2) if position is None else np.array(position, dtype=float)
        self.max_speed = max_speed
        rng = np.random.default_rng() if rng is None else rng
        self.velocity = (rng.random(2) - 0.5) * max_speed
        self.acceleration = np.zeros(2)
        self.behaviors = np.zeros((3, 2))
        self.color = '#FFFFFF' if color is None else color
        self.index = index

    def move(self, delta_time, width, height, border=100, out_decay=0.4,
             max_acceleration=150.0):
        # Soft steering near edges, followed by a final containment safeguard.
        border = min(border, (width - 2 * self.size) / 2,
                     (height - 2 * self.size) / 2)
        for axis, extent in enumerate((width, height)):
            lower = self.size + border
            upper = extent - self.size - border
            if self.position[axis] < lower:
                self.acceleration[axis] += (lower - self.position[axis]) / out_decay
            elif self.position[axis] > upper:
                self.acceleration[axis] += (upper - self.position[axis]) / out_decay
        magnitude = math.hypot(*self.acceleration)
        if magnitude > max_acceleration:
            self.acceleration *= max_acceleration / magnitude
        self.velocity += self.acceleration * delta_time
        speed = math.hypot(*self.velocity)
        if speed > self.max_speed:
            self.velocity *= self.max_speed / speed
        self.position += self.velocity * delta_time
        for axis, extent in enumerate((width, height)):
            if self.position[axis] < self.size:
                self.position[axis] = self.size
                self.velocity[axis] = abs(self.velocity[axis])
            elif self.position[axis] > extent - self.size:
                self.position[axis] = extent - self.size
                self.velocity[axis] = -abs(self.velocity[axis])

    def distance_to(self, other):
        return math.hypot(*(self.position - other.position))


class SpatialGrid:
    """Cells match the largest behavior radius: only nine cells are queried."""

    def __init__(self, boids, cell_size):
        self.cell_size = cell_size
        self.cells = defaultdict(list)
        for boid in boids:
            self.cells[self.key(boid.position)].append(boid)

    def key(self, position):
        return (math.floor(position[0] / self.cell_size),
                math.floor(position[1] / self.cell_size))

    def neighbors(self, boid):
        x, y = self.key(boid.position)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                yield from self.cells.get((x + dx, y + dy), ())



class ForceControls:
    """Mouse sliders share the same live weights as the keyboard controls."""

    FIELDS = (('sep_scalar', 'Separation [1/2]', COLORS[0]),
              ('ali_scalar', 'Alignment [3/4]', COLORS[1]),
              ('coh_scalar', 'Cohesion [5/6]', COLORS[2]))

    def __init__(self, args):
        self.args = args
        self.panel = pygame.Rect(WIDTH - 312, 10, 300, 274)
        self.tracks = [pygame.Rect(self.panel.x + 16, self.panel.y + 96 + i * 64,
                                   self.panel.width - 32, 8) for i in range(3)]
        self.maximums = [max(2.0, math.ceil(getattr(args, name)))
                         for name, _, _ in self.FIELDS]
        self.active = None

    def sync_ranges(self):
        # Preserve larger command-line/keyboard values without silently clamping.
        for i, (name, _, _) in enumerate(self.FIELDS):
            self.maximums[i] = max(self.maximums[i], math.ceil(getattr(self.args, name)))

    def set_from_mouse(self, x):
        track = self.tracks[self.active]
        fraction = min(1.0, max(0.0, (x - track.left) / track.width))
        name = self.FIELDS[self.active][0]
        setattr(self.args, name, round(fraction * self.maximums[self.active], 2))

    def handle_event(self, event):
        if event.type == pygame.WINDOWFOCUSLOST:
            self.active = None
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.sync_ranges()
            for i, track in enumerate(self.tracks):
                if track.inflate(16, 24).collidepoint(event.pos):
                    self.active = i
                    self.set_from_mouse(event.pos[0])
                    return True
        elif event.type == pygame.MOUSEMOTION and self.active is not None:
            if not event.buttons[0]:
                self.active = None
            else:
                self.set_from_mouse(event.pos[0])
            return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.active is not None:
                self.set_from_mouse(event.pos[0])
                self.active = None
                return True
        return False

    def draw(self, window, font):
        self.sync_ranges()
        pygame.draw.rect(window, '#151C28', self.panel, border_radius=10)
        pygame.draw.rect(window, '#354156', self.panel, width=1, border_radius=10)
        window.blit(font.render('Force weights', True, '#FFFFFF'),
                    (self.panel.x + 16, self.panel.y + 12))
        window.blit(font.render('Click or drag to adjust', True, '#AAB6C8'),
                    (self.panel.x + 16, self.panel.y + 38))
        for i, ((name, label, color), track) in enumerate(zip(self.FIELDS, self.tracks)):
            value = getattr(self.args, name)
            window.blit(font.render(label, True, color), (track.left, track.y - 26))
            text = font.render(f'{value:.2f}', True, '#FFFFFF')
            window.blit(text, (track.right - text.get_width(), track.y - 26))
            pygame.draw.rect(window, '#354156', track, border_radius=4)
            fraction = min(1.0, max(0.0, value / self.maximums[i]))
            knob_x = track.left + round(fraction * track.width)
            if knob_x > track.left:
                fill = pygame.Rect(track.left, track.top, knob_x - track.left, track.height)
                pygame.draw.rect(window, color, fill, border_radius=4)
            pygame.draw.circle(window, color, (knob_x, track.centery),
                               9 if self.active == i else 7)
            low = font.render('0', True, '#AAB6C8')
            high = font.render(f'{self.maximums[i]:g}', True, '#AAB6C8')
            window.blit(low, (track.left, track.bottom + 5))
            window.blit(high, (track.right - high.get_width(), track.bottom + 5))


def calculate_acceleration(boid, neighbors, args):
    """Return separation, alignment and cohesion without changing any state."""
    vectors = np.zeros((3, 2))
    counts = np.zeros(3, dtype=int)
    radius = max(args.sep_radius, args.ali_radius, args.coh_radius)
    for other in neighbors:
        if other is boid:
            continue
        offset = boid.position - other.position
        squared_distance = float(offset @ offset)
        if squared_distance >= radius * radius:
            continue
        dist = math.sqrt(squared_distance)
        if dist < args.sep_radius:
            if dist > 1e-9:
                # Inverse-distance repulsion, capped to avoid singular forces.
                repulsion = offset / dist * min(
                    args.sep_radius ** 2 / dist, args.max_acceleration)
            else:
                # Stable opposite directions for an exactly overlapping pair.
                first = (boid.index, id(boid)) < (other.index, id(other))
                repulsion = np.array([1.0 if first else -1.0, 0.0]) * args.max_acceleration
            weight = 1 - dist / args.sep_radius if args.smooth else 1.0
            vectors[0] += repulsion * weight
            counts[0] += 1
        if dist < args.ali_radius:
            weight = 1 - dist / args.ali_radius if args.smooth else 1.0
            vectors[1] += (other.velocity - boid.velocity) * weight
            counts[1] += 1
        if dist < args.coh_radius:
            weight = 1 - dist / args.coh_radius if args.smooth else 1.0
            vectors[2] -= offset * weight
            counts[2] += 1
    for i, scalar in enumerate((args.sep_scalar, args.ali_scalar, args.coh_scalar)):
        if counts[i]:
            vectors[i] *= scalar / counts[i]
    return vectors


def create_boids(args):
    rng = np.random.default_rng(args.seed)
    return [Boid(args.boid_size,
                 rng.uniform([args.boid_size] * 2,
                             [WIDTH - args.boid_size, HEIGHT - args.boid_size]),
                 args.max_speed, '#FF00FF' if i == 0 else '#FFFFFF', rng, i)
            for i in range(args.boids_num)]


def update(boids, args):
    grid = SpatialGrid(boids, max(args.sep_radius, args.ali_radius, args.coh_radius))
    # Every force reads the same snapshot; movement starts only after this loop.
    for boid in boids:
        boid.behaviors = calculate_acceleration(boid, grid.neighbors(boid), args)
        boid.acceleration = boid.behaviors.sum(axis=0)
    for boid in boids:
        boid.move(args.delta_time, WIDTH, HEIGHT,
                  max_acceleration=args.max_acceleration)


def draw(window, boids, args, font, paused, show_debug, fps, controls=None):
    window.fill('#000000')
    if show_debug and boids:
        reference = boids[0]
        for color, radius, vector in zip(
                COLORS, (args.sep_radius, args.ali_radius, args.coh_radius),
                reference.behaviors):
            pygame.draw.circle(window, color, reference.position, radius, 1)
            pygame.draw.line(window, color, reference.position,
                             reference.position + vector * 0.5, 2)
    for boid in boids:
        pygame.draw.circle(window, boid.color, boid.position, boid.size, 2)
        if show_debug:
            pygame.draw.line(window, boid.color, boid.position,
                             boid.position + boid.acceleration * 0.5, 1)
    lines = [
        f"{'PAUSED' if paused else 'RUNNING'} | Boids: {len(boids)} | FPS: {fps:.0f}",
        'Space: pause | R: reset | D: debug | Esc: quit',
    ]
    for i, line in enumerate(lines):
        text = font.render(line, True, '#FFFFFF', '#111111')
        window.blit(text, (10, 10 + i * 22))
    if controls is not None:
        controls.draw(window, font)
    pygame.display.flip()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smooth', action='store_true', help='Fade influence near each radius')
    parser.add_argument('--boids_num', type=int, default=50)
    parser.add_argument('--boid_size', type=int, default=8)
    parser.add_argument('--max_speed', type=float, default=200.0, help='Pixels per second')
    parser.add_argument('--max_acceleration', type=float, default=250.0, help='Pixels per second squared')
    parser.add_argument('--delta_time', type=float, default=1 / 60, help='Fixed simulation step in seconds (max 0.05)')
    parser.add_argument('--fps', type=int, default=60, help='Rendering frame rate cap')
    parser.add_argument('--sep_radius', type=float, default=22)
    parser.add_argument('--ali_radius', type=float, default=70)
    parser.add_argument('--coh_radius', type=float, default=110)
    parser.add_argument('--sep_scalar', type=float, default=0.44)
    parser.add_argument('--ali_scalar', type=float, default=0.8)
    parser.add_argument('--coh_scalar', type=float, default=0.09)
    parser.add_argument('--seed', type=int, default=603)
    args = parser.parse_args(argv)
    for name in ('boids_num', 'boid_size', 'max_speed', 'max_acceleration',
                 'delta_time', 'fps', 'sep_radius', 'ali_radius', 'coh_radius'):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f'--{name} must be finite and greater than zero')
    for name in ('sep_scalar', 'ali_scalar', 'coh_scalar'):
        value = getattr(args, name)
        if not math.isfinite(value) or value < 0:
            parser.error(f'--{name} must be finite and nonnegative')
    if args.boid_size * 2 >= min(WIDTH, HEIGHT):
        parser.error('--boid_size must be smaller than half the window dimensions')
    if args.delta_time > 0.05:
        parser.error('--delta_time must be at most 0.05 seconds')
    if args.seed < 0:
        parser.error('--seed must be nonnegative')
    return args


def main(argv=None):
    args = parse_args(argv)
    pygame.init()
    try:
        window = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Boids')
        font = pygame.font.Font(None, 22)
        clock = pygame.time.Clock()
        boids = create_boids(args)
        controls = ForceControls(args)
        running, paused, show_debug = True, False, True
        accumulator = 0.0
        weight_keys = {
            pygame.K_1: ('sep_scalar', -0.05), pygame.K_2: ('sep_scalar', 0.05),
            pygame.K_3: ('ali_scalar', -0.05), pygame.K_4: ('ali_scalar', 0.05),
            pygame.K_5: ('coh_scalar', -0.05), pygame.K_6: ('coh_scalar', 0.05),
        }
        while running:
            elapsed = min(clock.tick(args.fps) / 1000.0, 0.25)
            for event in pygame.event.get():
                controls.handle_event(event)
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        accumulator = 0.0
                        elapsed = 0.0
                    elif event.key == pygame.K_r:
                        boids = create_boids(args)
                        accumulator = 0.0
                        elapsed = 0.0
                    elif event.key == pygame.K_d:
                        show_debug = not show_debug
                    elif event.key in weight_keys:
                        name, change = weight_keys[event.key]
                        setattr(args, name, max(0.0, round(getattr(args, name) + change, 4)))
            if not running:
                break
            if not paused:
                # Bound catch-up work so an overloaded window stays responsive.
                accumulator = min(accumulator + elapsed, args.delta_time * 8)
                while accumulator >= args.delta_time:
                    update(boids, args)
                    accumulator -= args.delta_time
            draw(window, boids, args, font, paused, show_debug, clock.get_fps(), controls)
    finally:
        pygame.quit()


if __name__ == '__main__':
    main()
