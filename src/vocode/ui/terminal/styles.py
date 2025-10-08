from __future__ import annotations
from prompt_toolkit.styles import Style


def get_pt_style() -> Style:
    """
    Returns a prompt_toolkit Style object with all defined styles.
    """
    return Style.from_dict(
        {
            # Toolbar styles
            "bottom-toolbar": "bg:ansiblack fg:ansiwhite",
            "toolbar": "bg:ansiblack fg:ansiwhite",
            "toolbar.wf": "bg:ansiblack fg:ansigreen bold",
            "toolbar.node": "bg:ansiblack fg:ansigreen",
            # Prompt styles
            "prompt": "fg:#FFD700",
            # Banner styles
            "banner.l1": "fg:ansimagenta",
            "banner.l2": "fg:ansiblue",
            "banner.l3": "fg:ansicyan",
        }
    )
