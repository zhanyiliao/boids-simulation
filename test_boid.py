"""Behavioral and headless integration checks; run with python -m unittest -v."""
import contextlib
import io
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

import numpy as np
import pygame
import boid


class SimulationTests(unittest.TestCase):
    def setUp(self):
        self.args = boid.parse_args([])

    def test_grid_matches_full_neighbor_search(self):
        flock = boid.create_boids(boid.parse_args(['--boids_num', '150']))
        # Include cell edges, negative coordinates and an exact overlap.
        for item, point in zip(flock, [(0, 0), (-1, 0), (110, 110), (110, 110)]):
            item.position[:] = point
        for smooth in (False, True):
            self.args.smooth = smooth
            grid = boid.SpatialGrid(flock, 110)
            for item in flock:
                np.testing.assert_allclose(
                    boid.calculate_acceleration(item, grid.neighbors(item), self.args),
                    boid.calculate_acceleration(item, flock, self.args), atol=1e-12)

    def test_update_is_independent_of_iteration_order(self):
        forward = boid.create_boids(self.args)
        reverse = list(reversed(boid.create_boids(self.args)))
        for _ in range(10):
            boid.update(forward, self.args)
            boid.update(reverse, self.args)
        for a, b in zip(forward, reversed(reverse)):
            np.testing.assert_allclose(a.position, b.position, atol=1e-10)
            np.testing.assert_allclose(a.velocity, b.velocity, atol=1e-10)

    def test_alignment_vanishes_when_velocities_match(self):
        a, b = boid.create_boids(boid.parse_args(['--boids_num', '2']))
        b.position = a.position + [10, 0]
        b.velocity = a.velocity.copy()
        for smooth in (False, True):
            self.args.smooth = smooth
            result = boid.calculate_acceleration(a, [a, b], self.args)
            np.testing.assert_array_equal(result[1], [0, 0])

    def test_nearer_neighbor_has_stronger_repulsion(self):
        a, b = boid.create_boids(boid.parse_args(['--boids_num', '2']))
        b.position = a.position + [10, 0]
        far = boid.calculate_acceleration(a, [b], self.args)[0, 0]
        b.position = a.position + [5, 0]
        near = boid.calculate_acceleration(a, [b], self.args)[0, 0]
        self.assertLess(near, far)
        self.assertLess(far, 0)

    def test_overlap_separates_in_opposite_directions(self):
        flock = boid.create_boids(boid.parse_args(['--boids_num', '2']))
        for item in flock:
            item.position[:] = [500, 300]
            item.velocity[:] = 0
        boid.update(flock, self.args)
        self.assertGreater(flock[0].distance_to(flock[1]), 0)
        np.testing.assert_allclose(flock[0].velocity, -flock[1].velocity)

    def test_seed_reproduces_initial_state(self):
        for a, b in zip(boid.create_boids(self.args), boid.create_boids(self.args)):
            np.testing.assert_array_equal(a.position, b.position)
            np.testing.assert_array_equal(a.velocity, b.velocity)

    def test_long_run_respects_limits_and_stays_finite(self):
        self.args = boid.parse_args(['--boids_num', '20', '--smooth'])
        flock = boid.create_boids(self.args)
        for _ in range(300):
            boid.update(flock, self.args)
            for item in flock:
                self.assertTrue(np.isfinite(item.position).all())
                self.assertLessEqual(np.linalg.norm(item.velocity), self.args.max_speed + 1e-9)
                self.assertLessEqual(np.linalg.norm(item.acceleration), self.args.max_acceleration + 1e-9)
                self.assertTrue((item.position >= item.size).all())
                self.assertTrue((item.position <= np.array([boid.WIDTH, boid.HEIGHT]) - item.size).all())

    def test_boundary_contains_outward_motion(self):
        item = boid.Boid(8, [8, 8])
        item.velocity[:] = -100
        item.move(0.05, boid.WIDTH, boid.HEIGHT)
        self.assertTrue((item.position >= item.size).all())
        self.assertTrue((item.velocity >= 0).all())

    def test_invalid_arguments_are_rejected(self):
        for name, value in [('boids_num', '0'), ('max_speed', '-1'),
                            ('sep_radius', '0'), ('ali_scalar', 'nan'),
                            ('coh_scalar', '-1'), ('delta_time', '0.1666'),
                            ('fps', '0'), ('boid_size', '360'), ('seed', '-1')]:
            with self.subTest(name=name), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    boid.parse_args([f'--{name}', value])
                self.assertEqual(error.exception.code, 2)

    def test_fixed_steps_do_not_depend_on_render_rate(self):
        def run(frame_ms, frames):
            captured = []
            original = boid.create_boids
            def create(args):
                flock = original(args)
                captured.extend(flock)
                return flock
            events = [[] for _ in range(frames)] + [[pygame.event.Event(pygame.QUIT)]]
            clock = SimpleNamespace(tick=lambda fps: frame_ms, get_fps=lambda: 60)
            with patch.object(pygame.time, 'Clock', return_value=clock), \
                    patch.object(pygame.event, 'get', side_effect=events), \
                    patch.object(boid, 'draw'), patch.object(boid, 'create_boids', side_effect=create):
                boid.main(['--boids_num', '5'])
            return captured
        for a, b in zip(run(32, 30), run(16, 60)):
            np.testing.assert_allclose(a.position, b.position, atol=1e-10)
            np.testing.assert_allclose(a.velocity, b.velocity, atol=1e-10)

    def test_window_controls_and_rendering(self):
        key = lambda value: pygame.event.Event(pygame.KEYDOWN, key=value)
        events = [[], [key(pygame.K_SPACE)], [key(pygame.K_d), key(pygame.K_2)],
                  [key(pygame.K_r)], [key(pygame.K_SPACE)], [], [key(pygame.K_ESCAPE)]]
        clock = SimpleNamespace(tick=lambda fps: 20, get_fps=lambda: 50)
        with patch.object(pygame.time, 'Clock', return_value=clock), \
                patch.object(pygame.event, 'get', side_effect=events), \
                patch.object(boid, 'draw', wraps=boid.draw) as rendered:
            boid.main(['--boids_num', '5'])
        self.assertEqual(rendered.call_count, 6)
        self.assertTrue(rendered.call_args_list[1].args[4])
        self.assertFalse(rendered.call_args_list[2].args[5])
        self.assertFalse(rendered.call_args_list[-1].args[4])
        self.assertAlmostEqual(rendered.call_args.args[2].sep_scalar, 0.49)
        self.assertFalse(pygame.get_init())


class ForceControlsTests(unittest.TestCase):
    def setUp(self):
        self.args = boid.parse_args([])
        self.controls = boid.ForceControls(self.args)

    def test_click_drag_and_release_clamp_each_weight(self):
        for i, (name, _, _) in enumerate(self.controls.FIELDS):
            with self.subTest(name=name):
                track = self.controls.tracks[i]
                before = vars(self.args).copy()
                self.controls.handle_event(pygame.event.Event(
                    pygame.MOUSEBUTTONDOWN, button=1,
                    pos=(track.left + track.width // 4, track.centery)))
                self.assertAlmostEqual(getattr(self.args, name), 0.5)
                self.controls.handle_event(pygame.event.Event(
                    pygame.MOUSEMOTION, pos=(track.right + 200, track.centery),
                    buttons=(1, 0, 0)))
                self.assertEqual(getattr(self.args, name), 2)
                self.controls.handle_event(pygame.event.Event(
                    pygame.MOUSEBUTTONUP, button=1,
                    pos=(track.left - 200, track.centery)))
                self.assertEqual(getattr(self.args, name), 0)
                self.assertIsNone(self.controls.active)
                for other in ('sep_scalar', 'ali_scalar', 'coh_scalar'):
                    if other != name:
                        self.assertEqual(getattr(self.args, other), before[other])
                self.controls.handle_event(pygame.event.Event(
                    pygame.MOUSEMOTION, pos=track.center, buttons=(0, 0, 0)))
                self.assertEqual(getattr(self.args, name), 0)

    def test_focus_loss_or_missing_button_cancels_drag(self):
        track = self.controls.tracks[0]
        for cancel in (pygame.event.Event(pygame.WINDOWFOCUSLOST),
                       pygame.event.Event(pygame.MOUSEMOTION,
                                          pos=track.center, buttons=(0, 0, 0))):
            self.controls.handle_event(pygame.event.Event(
                pygame.MOUSEBUTTONDOWN, button=1, pos=track.center))
            self.controls.handle_event(cancel)
            self.assertIsNone(self.controls.active)
        before = self.args.sep_scalar
        for button, position in ((3, track.center), (1, (0, 0))):
            self.controls.handle_event(pygame.event.Event(
                pygame.MOUSEBUTTONDOWN, button=button, pos=position))
            self.assertIsNone(self.controls.active)
            self.assertEqual(self.args.sep_scalar, before)

    def test_ranges_preserve_larger_command_line_and_keyboard_values(self):
        args = boid.parse_args(['--sep_scalar', '3.5'])
        controls = boid.ForceControls(args)
        self.assertEqual(controls.maximums[0], 4)
        self.assertEqual(args.sep_scalar, 3.5)
        args.sep_scalar = 4.05
        controls.sync_ranges()
        self.assertEqual(controls.maximums[0], 5)
        self.assertEqual(args.sep_scalar, 4.05)

    def test_mouse_input_changes_live_simulation_and_works_while_paused(self):
        sep, ali, _ = self.controls.tracks
        events = [
            [pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1, pos=sep.center)],
            [pygame.event.Event(pygame.MOUSEMOTION, buttons=(1, 0, 0),
                                pos=(sep.left + sep.width * 3 // 4, sep.centery))],
            [pygame.event.Event(pygame.MOUSEBUTTONUP, button=1,
                                pos=(sep.right + 100, sep.centery))],
            [pygame.event.Event(pygame.KEYDOWN, key=pygame.K_SPACE),
             pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=1,
                                pos=(ali.left, ali.centery))],
            [pygame.event.Event(pygame.MOUSEBUTTONUP, button=1,
                                pos=(ali.left, ali.centery))],
            [pygame.event.Event(pygame.QUIT)],
        ]
        weights_used = []
        original_update = boid.update
        def update(flock, args):
            weights_used.append(args.sep_scalar)
            original_update(flock, args)
        clock = SimpleNamespace(tick=lambda fps: 20, get_fps=lambda: 50)
        with patch.object(pygame.time, 'Clock', return_value=clock), \
                patch.object(pygame.event, 'get', side_effect=events), \
                patch.object(boid, 'update', side_effect=update), \
                patch.object(boid, 'draw', wraps=boid.draw) as rendered:
            boid.main(['--boids_num', '5'])
        self.assertEqual(weights_used, [1.0, 1.5, 2.0])
        args = rendered.call_args.args[2]
        self.assertEqual(args.ali_scalar, 0)
        self.assertTrue(rendered.call_args.args[4])
        self.assertEqual(rendered.call_count, 5)


if __name__ == '__main__':
    unittest.main()
