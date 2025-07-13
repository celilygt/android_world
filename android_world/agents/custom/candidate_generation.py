# Copyright 2024 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Candidate Action Generation for V-DROID Agent.

This module implements the "Action Extractor" component from the V-DROID paper,
which generates all possible candidate actions from the current screen state
using both structured UI data and visual analysis.
"""

import tempfile
import os
from typing import List, Dict, Any, Optional
import xml.etree.ElementTree as ET

import cv2
import easyocr
import numpy as np
from PIL import Image

from android_world.env import interface
from android_world.env import representation_utils
import re


def parse_goal_for_context(goal: str) -> Dict[str, Any]:
    """Parse the goal string to extract relevant context information.
    
    Args:
        goal: The task goal string.
        
    Returns:
        Dictionary containing extracted context information.
    """
    if not goal:
        return {}
    
    context = {}
    goal_lower = goal.lower()
    
    # Extract names (looking for patterns like "Create a new contact for <name>")
    name_patterns = [
        r'contact for ([A-Za-z]+(?:\s+[A-Za-z]+)*)',  # "contact for John Doe"
        r'add contact ([A-Za-z]+(?:\s+[A-Za-z]+)*)',   # "add contact John Doe"
        r'create.*contact.*for ([A-Za-z]+(?:\s+[A-Za-z]+)*)',  # "create new contact for John Doe"
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, goal, re.IGNORECASE)
        if match:
            full_name = match.group(1).strip()
            context['full_name'] = full_name
            
            # Split into first and last name
            name_parts = full_name.split()
            if len(name_parts) >= 2:
                context['first_name'] = name_parts[0]
                context['last_name'] = ' '.join(name_parts[1:])
            else:
                context['first_name'] = full_name
            break
    
    # Extract phone numbers (looking for patterns like +1234567890 or (123) 456-7890)
    phone_patterns = [
        r'(\+?\d{10,15})',  # +1234567890 or 1234567890
        r'(\+?\d{1,3}[-.\s]?\d{3}[-.\s]?\d{3}[-.\s]?\d{4})',  # Various formats
        r'number is ([+\d\s\-\(\)]+)',  # "number is +1234567890"
        r'phone.*?([+\d\s\-\(\)]{10,})',  # "phone number +1234567890"
    ]
    
    for pattern in phone_patterns:
        match = re.search(pattern, goal, re.IGNORECASE)
        if match:
            phone = match.group(1).strip()
            # Clean up phone number (remove spaces, dashes, parentheses except +)
            phone_clean = re.sub(r'[^\d+]', '', phone)
            if len(phone_clean) >= 10:  # Valid phone number
                context['phone'] = phone_clean
            break
    
    # Extract email patterns
    email_match = re.search(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', goal)
    if email_match:
        context['email'] = email_match.group(1)
    
    return context


def generate_candidate_actions(state: interface.State, goal: str = None) -> List[Dict[str, Any]]:
    """Generate all possible candidate actions from the current state.
    
    This is the main function that combines structured UI data extraction
    and visual analysis to produce a comprehensive list of candidate actions.
    
    Args:
        state: The current environment state containing pixels and ui_elements.
        goal: The task goal string to extract context-specific information.
        
    Returns:
        List of candidate action dictionaries, each containing:
        - action_type: The type of action (click, input_text, scroll, etc.)
        - index: UI element index (for structured actions)
        - x, y: Coordinates (for visual actions)  
        - text: Associated text (if any)
        - Additional metadata
    """
    candidates = []
    
    # Parse goal for relevant information
    goal_context = parse_goal_for_context(goal) if goal else {}
    
    # 1. Extract candidates from structured UI elements
    ui_candidates = extract_candidates_from_ui_elements(state.ui_elements, goal_context)
    candidates.extend(ui_candidates)
    
    # 2. Extract candidates from visual analysis
    try:
        visual_candidates = extract_candidates_from_image(state.pixels)
        candidates.extend(visual_candidates)
    except Exception as e:
        print(f"Warning: Visual candidate extraction failed: {e}")
        # Continue without visual candidates if image processing fails
    
    # 3. Add UI-independent actions (from V-DROID paper Section 3.1)
    ui_independent_candidates = [
        {'action_type': 'navigate_home'},
        {'action_type': 'navigate_back'}, 
        {'action_type': 'status', 'goal_status': 'complete'},
        {'action_type': 'wait'},
    ]
    candidates.extend(ui_independent_candidates)
    
    # 4. Filter out candidates with None or empty text
    filtered_candidates = []
    for candidate in candidates:
        text = candidate.get('text')
        # Keep candidates that have meaningful text or are non-visual actions
        if (text and text.strip() and text.lower() != 'none') or candidate.get('action_type') in ['navigate_home', 'navigate_back', 'status', 'wait', 'open_app']:
            filtered_candidates.append(candidate)
    
    # 5. Merge and deduplicate candidates
    deduplicated_candidates = deduplicate_candidates(filtered_candidates)
    
    return deduplicated_candidates


def extract_candidates_from_ui_elements(ui_elements: List[representation_utils.UIElement], goal_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Extract candidate actions from structured UI elements.
    
    Based on the extract_candidates_from_xml.py function but adapted for
    AndroidWorld's UIElement objects instead of raw XML.
    
    Args:
        ui_elements: List of UIElement objects from the current state.
        goal_context: Dictionary containing extracted goal context information.
        
    Returns:
        List of candidate action dictionaries.
    """
    candidates = []
    
    for index, element in enumerate(ui_elements):
        # Extract clickable elements
        if getattr(element, 'is_clickable', False):
            candidates.append({
                'action_type': 'click',
                'index': index,
                'text': getattr(element, 'text', None),
                'resource_id': getattr(element, 'resource_id', None),
                'bounds': getattr(element, 'bbox', None),
                'content_desc': getattr(element, 'content_description', None),
            })
        
        # Extract scrollable elements  
        if getattr(element, 'is_scrollable', False):
            for direction in ['up', 'down', 'left', 'right']:
                candidates.append({
                    'action_type': 'scroll',
                    'index': index,
                    'direction': direction,
                    'text': getattr(element, 'text', None),
                    'resource_id': getattr(element, 'resource_id', None),
                })
        
        # Extract long-pressable elements
        if getattr(element, 'is_long_clickable', False):
            candidates.append({
                'action_type': 'long_press',
                'index': index,
                'text': getattr(element, 'text', None),
                'resource_id': getattr(element, 'resource_id', None),
            })
        
        # Extract input-capable elements (editable text fields)
        if getattr(element, 'is_editable', False) or \
           (hasattr(element, 'class_name') and 'EditText' in str(element.class_name)):
            # For input fields, we can generate candidates with goal-aware input patterns
            input_candidates = generate_input_text_candidates(index, element, goal_context)
            candidates.extend(input_candidates)
    
    return candidates


def generate_input_text_candidates(index: int, element: representation_utils.UIElement, goal_context: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """Generate input text candidates for editable elements.
    
    Args:
        index: The UI element index.
        element: The UIElement object.
        goal_context: Dictionary containing extracted goal context information.
        
    Returns:
        List of input text candidate dictionaries.
    """
    candidates = []
    goal_context = goal_context or {}
    
    # Common input patterns based on element context
    text_hint = getattr(element, 'text', '') or getattr(element, 'content_description', '')
    resource_id = getattr(element, 'resource_id', '')
    
    # Generate context-aware input suggestions
    if text_hint:
        hint_lower = text_hint.lower()
        
        if any(word in hint_lower for word in ['search', 'find', 'query']):
            input_texts = ['android', 'app', 'search query']
        elif any(word in hint_lower for word in ['email', 'mail']):
            input_texts = []
            if 'email' in goal_context:
                input_texts.append(goal_context['email'])
            input_texts.extend(['test@example.com', 'user@gmail.com'])
        elif any(word in hint_lower for word in ['password', 'pwd']):
            input_texts = ['password123', 'test123']
        elif any(word in hint_lower for word in ['first', 'name']) and 'first' in hint_lower:
            # First name field
            input_texts = []
            if 'first_name' in goal_context:
                input_texts.append(goal_context['first_name'])
            input_texts.extend(['John', 'TestUser'])
        elif any(word in hint_lower for word in ['last', 'name']) and 'last' in hint_lower:
            # Last name field  
            input_texts = []
            if 'last_name' in goal_context:
                input_texts.append(goal_context['last_name'])
            input_texts.extend(['Doe', 'TestUser'])
        elif any(word in hint_lower for word in ['name', 'user']):
            # General name field
            input_texts = []
            if 'full_name' in goal_context:
                input_texts.append(goal_context['full_name'])
            if 'first_name' in goal_context:
                input_texts.append(goal_context['first_name'])
            input_texts.extend(['John Doe', 'TestUser'])
        elif any(word in hint_lower for word in ['phone', 'number']):
            # Phone number field
            input_texts = []
            if 'phone' in goal_context:
                input_texts.append(goal_context['phone'])
            input_texts.extend(['1234567890', '+1234567890'])
        else:
            input_texts = ['test', 'sample text']
    else:
        # Default input texts
        input_texts = ['test', 'sample text']
    
    # Remove duplicates while preserving order
    seen = set()
    unique_texts = []
    for text in input_texts:
        if text and text not in seen:
            seen.add(text)
            unique_texts.append(text)
    
    for text in unique_texts:
        candidates.append({
            'action_type': 'input_text',
            'index': index,
            'text': text,
            'resource_id': resource_id,
        })
    
    return candidates


def extract_candidates_from_image(pixels: np.ndarray) -> List[Dict[str, Any]]:
    """Extract candidate actions from visual analysis of the screenshot.
    
    Based on the generate_candidates_from_image.py function but adapted for
    AndroidWorld's numpy array format.
    
    Args:
        pixels: RGB numpy array of the current screen.
        
    Returns:
        List of candidate action dictionaries.
    """
    candidates = []
    
    # Convert numpy array to OpenCV format
    img = cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    
    # Initialize OCR reader
    try:
        reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        results = reader.readtext(img)
    except Exception as e:
        print(f"OCR initialization failed: {e}")
        return candidates
    
    # OCR-based candidates
    for (bbox, text, confidence) in results:
        if confidence > 0.5:  # Only include confident detections
            # Filter out debugging/performance overlay text
            if _is_debugging_text(text):
                continue
                
            (top_left, top_right, bottom_right, bottom_left) = bbox
            x = int((top_left[0] + bottom_right[0]) / 2)
            y = int((top_left[1] + bottom_right[1]) / 2)
            element_type = guess_element_type(text, confidence, x, y)
            
            candidates.append({
                'action_type': 'click',
                'x': x,
                'y': y,
                'text': text,
                'confidence': round(confidence, 2),
                'element_type': element_type,
                'source': 'ocr'
            })
    
    # Skip contour-based candidates for now - they create too much noise
    # OCR-based candidates are more reliable and meaningful
    # TODO: Re-enable contour detection with better filtering if needed
    
    return candidates


def _is_debugging_text(text: str) -> bool:
    """Check if text appears to be debugging/performance overlay data.
    
    Args:
        text: The OCR-detected text.
        
    Returns:
        True if this looks like debugging text that should be filtered out.
    """
    if not text or len(text.strip()) == 0:
        return True
    
    text_lower = text.lower().strip()
    
    # Common debugging patterns
    debugging_patterns = [
        # Performance metrics
        'p:', 'dx:', 'dy:', 'xv:', 'yv:', 'prs:', 'pprs:', 'size:',
        # Numbers with colons/decimals (likely metrics)
        r'^\d+\.\d+$',  # Like "15.34"
        r'^[a-z]+:\s*\d',  # Like "P: 0"
        r'^\d+\s*/\s*\d+$',  # Like "0 / 1"
        # Very short fragments
        r'^[a-z]$',  # Single letters
        r'^\d+$',  # Just numbers
    ]
    
    import re
    for pattern in debugging_patterns:
        if pattern.startswith('r\''):  # Regex pattern
            if re.match(pattern[2:-1], text_lower):
                return True
        else:  # Simple string check
            if pattern in text_lower:
                return True
                
    # Filter very short text (likely noise)
    if len(text_lower) <= 2:
        return True
        
    return False


def guess_element_type(text: str, confidence: float, x: int, y: int) -> str:
    """Guess the type of UI element based on text content and position.
    
    Args:
        text: The OCR-detected text.
        confidence: OCR confidence score.
        x, y: Element position coordinates.
        
    Returns:
        String describing the likely element type.
    """
    text_lower = text.lower()
    
    if any(word in text_lower for word in ["search", "find", "query", "type here"]):
        return "search"
    elif any(word in text_lower for word in ["ok", "confirm", "login", "submit", "start", "continue"]):
        return "button"
    elif any(word in text_lower for word in ["cancel", "back", "close", "exit"]):
        return "cancel_button"
    elif any(word in text_lower for word in ["menu", "options", "settings"]):
        return "menu"
    elif confidence > 0.9 and len(text) <= 15:
        return "button"
    elif len(text) > 30:
        return "paragraph"
    elif text.strip():
        return "label"
    else:
        return "unknown"


def deduplicate_candidates(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate candidates based on action type and target.
    
    Args:
        candidates: List of candidate action dictionaries.
        
    Returns:
        Deduplicated list of candidates.
    """
    seen = set()
    deduplicated = []
    
    for candidate in candidates:
        # Create a signature for the candidate
        signature_parts = [candidate.get('action_type', '')]
        
        if 'index' in candidate:
            signature_parts.append(f"index:{candidate['index']}")
        if 'x' in candidate and 'y' in candidate:
            # Round coordinates to reduce minor variations
            x_rounded = round(candidate['x'] / 20) * 20  # Round to nearest 20 pixels
            y_rounded = round(candidate['y'] / 20) * 20
            signature_parts.append(f"coords:{x_rounded},{y_rounded}")
        if 'direction' in candidate:
            signature_parts.append(f"dir:{candidate['direction']}")
        if 'text' in candidate and candidate['text']:
            signature_parts.append(f"text:{candidate['text'][:20]}")  # First 20 chars
        
        signature = "|".join(signature_parts)
        
        if signature not in seen:
            seen.add(signature)
            deduplicated.append(candidate)
    
    return deduplicated


def clean_candidate_for_json_action(candidate: Dict[str, Any]) -> Dict[str, Any]:
    """Clean candidate dictionary to only include fields accepted by JSONAction.
    
    Args:
        candidate: Raw candidate action dictionary.
        
    Returns:
        Cleaned dictionary with only JSONAction-compatible fields.
    """
    # Valid JSONAction fields from the class definition
    valid_fields = {
        'action_type', 'index', 'x', 'y', 'text', 
        'direction', 'goal_status', 'app_name', 'keycode'
    }
    
    # Create cleaned candidate with only valid fields
    cleaned = {}
    for key, value in candidate.items():
        if key in valid_fields and value is not None:
            cleaned[key] = value
    
    return cleaned


def format_candidates_for_display(candidates: List[Dict[str, Any]]) -> str:
    """Format candidates for human-readable display.
    
    Args:
        candidates: List of candidate action dictionaries.
        
    Returns:
        Formatted string describing the candidates.
    """
    if not candidates:
        return "No candidates found."
    
    lines = [f"Found {len(candidates)} candidate actions:"]
    
    for i, candidate in enumerate(candidates):
        action_type = candidate.get('action_type', 'unknown')
        
        if action_type == 'click':
            if 'index' in candidate:
                text = candidate.get('text', 'no text')
                lines.append(f"  {i+1}. Click UI element {candidate['index']} ('{text}')")
            else:
                x, y = candidate.get('x', '?'), candidate.get('y', '?')
                text = candidate.get('text', 'no text')
                lines.append(f"  {i+1}. Click at ({x}, {y}) ('{text}')")
        elif action_type == 'input_text':
            text = candidate.get('text', '')
            lines.append(f"  {i+1}. Input '{text}' into element {candidate.get('index', '?')}")
        elif action_type == 'scroll':
            direction = candidate.get('direction', '?')
            lines.append(f"  {i+1}. Scroll {direction} on element {candidate.get('index', '?')}")
        else:
            lines.append(f"  {i+1}. {action_type.replace('_', ' ').title()}")
    
    return "\n".join(lines) 