"""Generate the gheim wordmark logo as PNG.

Renders the word "gheim" in bold sans-serif and overlays a black redaction
bar that covers the bulk of the x-height. The bar is sized so that:
  - the ascender of 'h' remains visible above the bar
  - the dot of 'i' remains visible above the bar
  - the descender of 'g' remains visible below the bar
  - thin slivers at top and bottom of e/i/m peek out
The word is readable only if you already know it should spell 'gheim'.

Usage:
    uv run --with pillow python assets/make_logo.py
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WORD = "gheim"
FONT_PATH = "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf"
FONT_SIZE = 256
PAD_X = 40
PAD_TOP = 16
PAD_BOTTOM = 16
BAR_TOP_FRAC = 0.49   # bar top, as a fraction of ascent below the canvas top.
                      # Larger value = bar starts lower = more letter visible on top.
BAR_BOTTOM_OFFSET = 12  # px above the baseline; larger = more bottom sliver of e/i/m visible.
OUT_PATH = Path(__file__).parent / "logo.png"


def main() -> None:
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)

    bbox = font.getbbox(WORD)
    text_w = bbox[2] - bbox[0]

    ascent, descent = font.getmetrics()
    img_w = text_w + 2 * PAD_X
    img_h = ascent + descent + PAD_TOP + PAD_BOTTOM
    baseline_y = PAD_TOP + ascent

    cap_height = ascent
    bar_top_y = PAD_TOP + int(cap_height * BAR_TOP_FRAC)
    bar_bottom_y = baseline_y - BAR_BOTTOM_OFFSET

    img = Image.new("RGBA", (img_w, img_h), (255, 255, 255, 0))
    draw = ImageDraw.Draw(img)
    draw.text((PAD_X, baseline_y), WORD, font=font, fill=(0, 0, 0, 255), anchor="ls")

    bar_left = PAD_X - 4
    bar_right = PAD_X + text_w + 4
    draw.rectangle(
        [(bar_left, bar_top_y), (bar_right, bar_bottom_y)],
        fill=(0, 0, 0, 255),
    )

    img.save(OUT_PATH, optimize=True)
    print(f"Wrote {OUT_PATH}  ({img_w}x{img_h})")
    print(f"  font:        {FONT_PATH}")
    print(f"  font_size:   {FONT_SIZE}")
    print(f"  text bbox:   {bbox}")
    print(f"  baseline:    y={baseline_y}")
    print(f"  bar:         y={bar_top_y} to y={bar_bottom_y}")


if __name__ == "__main__":
    main()
