"""
Screen Analyzer for V-DROID Agent

This module provides functions to analyze the current UI state for common
contextual situations that can help the agent make better decisions.
"""

from typing import List, Dict, Any

from android_world.env import representation_utils


def analyze_screen_for_context(ui_elements: List[representation_utils.UIElement]) -> List[str]:
    """Analyzes the UI elements to detect key contextual states.

    Args:
        ui_elements: A list of UIElement objects from the current screen.

    Returns:
        A list of strings describing the detected contexts (e.g., ["POPUP_DETECTED"]).
    """
    contexts = []

    if is_popup_dialog_present(ui_elements):
        contexts.append("POPUP_DETECTED")

    if is_keyboard_visible(ui_elements):
        contexts.append("KEYBOARD_VISIBLE")

    # Add more detectors here in the future (e.g., for loaders, tutorials)

    return contexts

def is_popup_dialog_present(ui_elements: List[representation_utils.UIElement]) -> bool:
    """Detects if a pop-up dialog is likely present on the screen.

    Heuristic: Looks for common dialog button text like 'Allow', 'Deny', 'OK',
    'Cancel' in clickable, distinct UI elements.
    """
    dialog_button_texts = {"allow", "deny", "ok", "cancel", "yes", "no", "got it", "continue"}
    buttons_found = 0

    for element in ui_elements:
        text = (element.text or "").lower()
        if element.is_clickable and text in dialog_button_texts:
            # Check if it's a standalone button and not part of a larger list
            if element.bbox and element.bbox.height < 200: # Simple heuristic for button size
                buttons_found += 1
    
    # A dialog usually has one or two of these buttons.
    return buttons_found > 0

def is_keyboard_visible(ui_elements: List[representation_utils.UIElement]) -> bool:
    """Detects if a keyboard is likely visible on the screen.

    Heuristic: Checks for the presence of a UI element with a resource ID
    common to Android keyboards.
    """
    keyboard_resource_ids = [
        "com.google.android.inputmethod.latin:id/keyboard_view", # Gboard
        "com.android.inputmethod.latin:id/keyboard_view" # AOSP Keyboard
    ]
    for element in ui_elements:
        if element.resource_id in keyboard_resource_ids:
            return True
    return False
