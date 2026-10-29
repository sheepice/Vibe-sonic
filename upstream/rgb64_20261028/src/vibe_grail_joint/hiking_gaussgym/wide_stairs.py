"""Evaluation-only doubled tread depth; riser height and lateral width unchanged."""
from ..velocity.uniform_probe20 import probe_tiles as original_probe_tiles


def probe_tiles():
    tiles = []
    for original in original_probe_tiles():
        tile = dict(original)
        tile['tread'] = 2 * original['tread']
        # Preserve the 1.2 m summit, including the last ascending tread.
        tile['plateau'] = original['plateau'] + original['tread'] - tile['tread']
        end = tile['start'] + tile['n'] * tile['tread']
        if tile['kind'] == 'up_down':
            end += tile['plateau'] + tile['n'] * tile['tread']
        tile['goal'] = end + .6
        assert tile['goal'] < 12., 'Goal must remain within the physical mesh'
        tiles.append(tile)
    return tuple(tiles)
