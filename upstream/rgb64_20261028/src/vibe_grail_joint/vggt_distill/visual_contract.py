"""Full-frame patch coverage for pixel-aligned Theia/VGGT supervision."""

PATCH_SIZE = 16


def validate_patch_coverage(height, width, grid):
    """Reject implicit bottom/right crops before full-frame teacher resizing."""
    if any(not isinstance(n, int) or isinstance(n, bool) or n <= 0 for n in (height, width)):
        raise ValueError('RGB height and width must be positive integers')
    if height % PATCH_SIZE or width % PATCH_SIZE:
        raise ValueError(
            f'Incomplete patch coverage for RGB {width}x{height}: Theia would discard '
            f'{height % PATCH_SIZE} bottom rows and {width % PATCH_SIZE} right columns, '
            'while VGGT supervises the full frame. Use dimensions divisible by 16.'
        )
    expected = (height // PATCH_SIZE, width // PATCH_SIZE)
    if tuple(grid) != expected:
        raise ValueError(f'Teacher/student patch grid mismatch: {tuple(grid)} != {expected}')
    return dict(rgb_shape_hwc=[height, width, 3], patch_grid=list(expected),
                patch_size=PATCH_SIZE, patch_count=expected[0] * expected[1],
                discarded_bottom_rows=0, discarded_right_columns=0)


def validate_training_visual_contract(settings):
    camera, distillation = settings['camera'], settings['distillation']
    contract = validate_patch_coverage(camera['height'], camera['width'], distillation['patch_grid'])
    teacher_height, teacher_width = distillation['teacher_shape']
    validate_patch_coverage(teacher_height, teacher_width,
                            (teacher_height // PATCH_SIZE, teacher_width // PATCH_SIZE))
    contract['teacher_shape_hw'] = [teacher_height, teacher_width]
    return contract
