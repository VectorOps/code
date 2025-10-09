from __future__ import annotations
from prompt_toolkit.styles import Style


def get_pt_style() -> Style:
    return Style.from_dict(
        {
            # Prompt Toolkit
            "bottom-toolbar": "noreverse bg:ansiblack fg:ansiwhite",
            "bottom-toolbar.text": "noreverse bg:ansiblack fg:ansiwhite",
            # Toolbar and prompt
            "toolbar": "",
            "toolbar.wf": "bold",
            "toolbar.node": "",
            # Prompt
            "prompt": "fg: #FFC000",
            # System messages
            "system.star": "ansigreen bold",
            "system.text": "ansiwhite",
            "system.highlight": "ansiwhite bold",
            # Banner (optional)
            "banner.l1": "ansicyan bold",
            "banner.l2": "ansiwhite bold",
            "banner.l3": "ansiblue bold",
            # Tool-call formatting
            "toolcall.name": "ansicyan bold",
            "toolcall.parameter": "ansiwhite",
            "toolcall.separator": "ansigray",
        }
    )
