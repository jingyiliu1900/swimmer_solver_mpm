"""Render simulation frames to a repo-friendly animated GIF.

GitHub renders animated GIFs inline in Markdown (so they show up directly in the
README), unlike MP4 which it will not embed. The examples therefore rasterise
their Matplotlib figures with :func:`figure_to_rgb` and assemble them into a GIF
with :func:`save_gif` -- no ffmpeg needed, just Pillow.
"""

from __future__ import annotations

import numpy as np


def figure_to_rgb(fig) -> np.ndarray:
    """Rasterise a Matplotlib figure (Agg backend) to an ``(H, W, 3)`` uint8 array."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[..., :3].copy()


def save_gif(frames, path: str, fps: int = 25, colors: int = 64) -> str:
    """Write a list of ``(H, W, 3)`` uint8 frames to an animated GIF via Pillow.

    Each frame is quantised to a ``colors``-entry palette (GIF is paletted
    anyway), which keeps the file small enough to commit and embed. Returns
    ``path``; loops forever, and ``disposal=2`` clears each frame so moving
    particles don't smear.
    """
    from PIL import Image

    if not frames:
        raise ValueError("no frames to write")
    images = [
        Image.fromarray(np.ascontiguousarray(f)).quantize(
            colors=colors, method=Image.Quantize.FASTOCTREE
        )
        for f in frames
    ]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=max(1, round(1000 / fps)),
        loop=0,
        optimize=True,
        disposal=2,
    )
    return path
