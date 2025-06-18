import xml.etree.ElementTree as ET
import sys

def extract_candidates_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    candidates = []
    for node in root.iter('node'):
        attrs = node.attrib
        if attrs.get('clickable') == 'true':
            candidates.append({
                "action_type": "click",
                "text": attrs.get('text'),
                "resource_id": attrs.get('resource-id'),
                "bounds": attrs.get('bounds')
            })
        if attrs.get('focusable') == 'true':
            candidates.append({
                "action_type": "focus",
                "text": attrs.get('text'),
                "resource_id": attrs.get('resource-id'),
                "bounds": attrs.get('bounds')
            })
        if attrs.get('editable') == 'true':
            candidates.append({
                "action_type": "input_text",
                "text": attrs.get('text'),
                "resource_id": attrs.get('resource-id'),
                "bounds": attrs.get('bounds')
            })
    return candidates

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract_candidates_from_xml.py <window_dump.xml>")
        sys.exit(1)
    xml_path = sys.argv[1]
    candidates = extract_candidates_from_xml(xml_path)
    for c in candidates:
        print(c)
