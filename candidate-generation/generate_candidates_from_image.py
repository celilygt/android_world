import cv2
import easyocr
import json
import sys
import os

def generate_candidates_from_image(image_file):
    reader = easyocr.Reader(['en', 'de'], gpu=False)
    img = cv2.imread(image_file)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {image_file}")

    h, w = img.shape[:2]
    results = reader.readtext(img)
    candidates = []

    # OCR candidates
    for (bbox, text, confidence) in results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        x = int((top_left[0] + bottom_right[0]) / 2)
        y = int((top_left[1] + bottom_right[1]) / 2)
        element_type = guess_element_type(text, confidence, x, y)
        candidates.append({
            "action_type": "click",
            "x": x,
            "y": y,
            "text": text,
            "confidence": round(confidence, 2),
            "element_type": element_type
        })

    # Contour candidates (for buttons/areas without text)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 30, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w_box, h_box = cv2.boundingRect(c)
        if w_box > 40 and h_box > 20 and w_box < int(w*0.8) and h_box < int(h*0.8):
            mid_x = x + w_box // 2
            mid_y = y + h_box // 2
            overlap = False
            for cand in candidates:
                if abs(cand["x"] - mid_x) < 20 and abs(cand["y"] - mid_y) < 20:
                    overlap = True
                    break
            if not overlap:
                candidates.append({
                    "action_type": "click",
                    "x": mid_x,
                    "y": mid_y,
                    "text": None,
                    "confidence": None,
                    "element_type": "maybe_button"
                })

    # Fallback: Search bar candidate (if none detected)
    if not any(c["element_type"] == "search" for c in candidates):
        approx_search_x = w // 2
        approx_search_y = int(h * 0.07)
        candidates.append({
            "action_type": "click",
            "x": approx_search_x,
            "y": approx_search_y,
            "text": "",
            "confidence": None,
            "element_type": "search"
        })

    return candidates

def guess_element_type(text, confidence, x, y):
    text_lower = text.lower()
    if any(word in text_lower for word in ["search", "suchen", "type here", "eingeben"]):
        return "search"
    elif any(word in text_lower for word in ["ok", "weiter", "login", "submit", "start"]):
        return "button"
    elif confidence > 0.9 and len(text) <= 15:
        return "button"
    elif len(text) > 20:
        return "paragraph"
    elif text.strip():
        return "label"
    else:
        return "unknown"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_candidates_from_image.py <screenshot.png>")
        sys.exit(1)
    image_file = sys.argv[1]
    if not os.path.exists(image_file):
        print(f"File not found: {image_file}")
        sys.exit(1)
    actions = generate_candidates_from_image(image_file)
    print(json.dumps(actions, indent=2, ensure_ascii=False))
