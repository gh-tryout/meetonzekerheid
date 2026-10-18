"""Formula field that inserts symbols at the caret."""

from pathlib import Path

import streamlit.components.v1 as components

_component = components.declare_component("formula_field", path=str(Path(__file__).parent))


def formula_input(text: str, generation: int) -> str:
    """Show the formula field. ``generation`` remounts it after loading a file."""
    value = _component(text=text, key=f"formula_{generation}", default=text)
    return text if not isinstance(value, str) else value
