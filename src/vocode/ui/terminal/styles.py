from __future__ import annotations
from prompt_toolkit.styles import Style


def get_pt_style() -> Style:
    """
    Returns a prompt_toolkit Style object with all defined styles.
    """
    return Style.from_dict(
        {
            # Toolbar styles
            "bottom-toolbar": "noreverse",
            "bottom-toolbar.text": "noreverse",
            "toolbar": "bg:#2E2E2E #F0F0F0",
            "toolbar.wf": "bg:#2E2E2E fg:ansiwhite bold",
            "toolbar.node": "bg:#2E2E2E fg:ansiwhite",
            # Prompt styles
            "prompt": "fg:#FFD700",
            # Banner styles
            "banner.l1": "fg:ansimagenta",
            "banner.l2": "fg:ansiblue",
            "banner.l3": "fg:ansicyan",
        }
    )
