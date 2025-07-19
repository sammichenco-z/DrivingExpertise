import os
import csv
import xml.etree.ElementTree as ET
from pathlib import Path

def process_file(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    csv_file = Path(xml_file).parent / (Path(xml_file).parent.name + '.csv')
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'name', 'width', 'height', 'label', 'source', 'occluded', 'xtl', 'ytl', 'xbr', 'ybr', 'z_order'])
        for image in root.findall('image'):
            image_id = image.get('id')
            image_name = image.get('name')
            image_width = image.get('width')
            image_height = image.get('height')
            for box in image.findall('box'):
                box_label = box.get('label')
                box_source = box.get('source')
                box_occluded = box.get('occluded')
                box_xtl = box.get('xtl')
                box_ytl = box.get('ytl')
                box_xbr = box.get('xbr')
                box_ybr = box.get('ybr')
                box_z_order = box.get('z_order')
                writer.writerow([image_id, image_name, image_width, image_height, box_label, box_source, box_occluded, box_xtl, box_ytl, box_xbr, box_ybr, box_z_order])

def main():
    directory = r'M:\Carla\carla\PythonAPI\examples\record_out\CVAT AOI'
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == 'annotations.xml':
                print("PROCESSING ", root)
                xml_file = os.path.join(root, file)
                process_file(xml_file)

if __name__ == '__main__':
    main()
