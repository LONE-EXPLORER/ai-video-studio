"""
app/captions/styles.py — Caption style presets for ASS subtitle generation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class CaptionStyle:
    """
    ASS subtitle style definition.

    Colour values use the ASS ABGR hex format: ``&H<alpha><blue><green><red>``.
    Alpha ``00`` = fully opaque, ``FF`` = fully transparent.
    """

    name: str
    font_name: str
    font_size: int
    primary_colour: str    # text colour
    secondary_colour: str  # karaoke pre-highlight colour
    outline_colour: str    # outline / border colour
    back_colour: str       # shadow / background box colour
    bold: int              # 1 = bold, 0 = normal
    italic: int            # 1 = italic, 0 = normal
    alignment: int         # ASS numpad alignment (2 = bottom-centre, 8 = top-centre)
    outline: float         # border width in pixels
    shadow: float          # shadow depth in pixels
    margin_v: int          # vertical margin in pixels

    def to_ass_line(self) -> str:
        """Return the ``Style:`` line for the [V4+ Styles] section."""
        return (
            f"Style: {self.name},"
            f"{self.font_name},{self.font_size},"
            f"{self.primary_colour},{self.secondary_colour},"
            f"{self.outline_colour},{self.back_colour},"
            f"{self.bold},{self.italic},0,0,"  # Underline, StrikeOut
            f"100,100,0,0,"                     # ScaleX, ScaleY, Spacing, Angle
            f"1,{self.outline},{self.shadow},"  # BorderStyle, Outline, Shadow
            f"{self.alignment},20,20,{self.margin_v},1"  # MarginL, MarginR, MarginV, Encoding
        )


# ---------------------------------------------------------------------------
# Style presets
# ---------------------------------------------------------------------------

DYNAMIC_STYLE = CaptionStyle(
    name="Dynamic",
    font_name="Arial",
    font_size=64,
    primary_colour="&H00FFFFFF",   # white text
    secondary_colour="&H0000D7FF", # yellow pre-highlight (karaoke)
    outline_colour="&H00000000",   # black outline
    back_colour="&H80000000",      # 50% transparent black shadow
    bold=1,
    italic=0,
    alignment=2,                   # bottom-centre
    outline=3.0,
    shadow=1.5,
    margin_v=100,
)

CLEAN_STYLE = CaptionStyle(
    name="Clean",
    font_name="Arial",
    font_size=52,
    primary_colour="&H00FFFFFF",   # white
    secondary_colour="&H00FFFFFF",
    outline_colour="&H00000000",   # black outline
    back_colour="&H00000000",
    bold=0,
    italic=0,
    alignment=2,                   # bottom-centre
    outline=2.0,
    shadow=1.0,
    margin_v=80,
)

MINIMAL_STYLE = CaptionStyle(
    name="Minimal",
    font_name="Arial",
    font_size=36,
    primary_colour="&H00FFFFFF",   # white
    secondary_colour="&H00FFFFFF",
    outline_colour="&H00000000",
    back_colour="&HAA000000",      # translucent dark background
    bold=0,
    italic=0,
    alignment=2,
    outline=1.0,
    shadow=0.0,
    margin_v=50,
)

DEFAULT_STYLE = CaptionStyle(
    name="Default",
    font_name="Arial",
    font_size=48,
    primary_colour="&H00FFFFFF",
    secondary_colour="&H00FFFFFF",
    outline_colour="&H00000000",
    back_colour="&H80000000",
    bold=0,
    italic=0,
    alignment=2,
    outline=2.0,
    shadow=1.0,
    margin_v=80,
)

# Registry
_STYLES: Dict[str, CaptionStyle] = {
    "dynamic": DYNAMIC_STYLE,
    "clean": CLEAN_STYLE,
    "minimal": MINIMAL_STYLE,
    "default": DEFAULT_STYLE,
    "highlighted": DYNAMIC_STYLE,  # alias
}


def get_style(name: str) -> CaptionStyle:
    """
    Return the :class:`CaptionStyle` for *name*, falling back to Default.

    Parameters
    ----------
    name:
        Style preset name (case-insensitive).
    """
    return _STYLES.get(name.lower(), DEFAULT_STYLE)


def get_ass_style_block(style_name: str) -> str:
    """
    Return the complete ASS ``Style:`` lines string for the given preset.

    Includes all relevant styles so the ASS file is self-contained.

    Parameters
    ----------
    style_name:
        Style preset name.
    """
    primary = get_style(style_name)
    # Always include all styles so the file is usable independently
    all_styles = [DYNAMIC_STYLE, CLEAN_STYLE, MINIMAL_STYLE, DEFAULT_STYLE]
    seen = set()
    lines = []
    for s in all_styles:
        if s.name not in seen:
            lines.append(s.to_ass_line())
            seen.add(s.name)
    return "\n".join(lines)
