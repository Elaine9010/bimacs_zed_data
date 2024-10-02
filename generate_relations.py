# This script read data from 3d_obj , generates spatial relations between objects and saves them to json files
# Usage: python generate_relations.py -input_folder <input_folder> -output_folder <output_folder>
# -input_folder our_derived_data/subject1/task_1_k_cooking
# -output_folder our_derived_data/subject1/task_1_k_cooking

import json
from typing import List, Dict, Any
import math
from typing import List, NamedTuple
import os
import argparse

class BoundingBox(NamedTuple):
    x0: float
    x1: float
    y0: float
    y1: float
    z0: float
    z1: float

class DetectedObject(NamedTuple):
    class_name: str
    bounding_box: BoundingBox
    past_bounding_box: BoundingBox

class Relations:
    def __init__(self):
        self.contact = False
        self.static_left_of = False
        self.static_right_of = False
        self.static_below = False
        self.static_above = False
        self.static_behind_of = False
        self.static_in_front_of = False
        self.static_inside = False
        self.static_surround = False
        self.dynamic_moving_together = False
        self.dynamic_halting_together = False
        self.dynamic_fixed_moving_together = False
        self.dynamic_getting_close = False
        self.dynamic_moving_apart = False
        self.dynamic_stable = False

def is_colliding(a: BoundingBox, b: BoundingBox) -> bool:
    return (a.x0 <= b.x1 and a.x1 >= b.x0 and
            a.y0 <= b.y1 and a.y1 >= b.y0 and
            a.z0 <= b.z1 and a.z1 >= b.z0)

def distance_between(a: BoundingBox, b: BoundingBox) -> float:
    cax = (a.x1 + a.x0) / 2
    cay = (a.y1 + a.y0) / 2
    caz = (a.z1 + a.z0) / 2
    cbx = (b.x1 + b.x0) / 2
    cby = (b.y1 + b.y0) / 2
    cbz = (b.z1 + b.z0) / 2
    
    distance = math.sqrt((cax - cbx)**2 + (cay - cby)**2 + (caz - cbz)**2)
    assert distance >= 0, f"Calculated distance cannot be less than zero (is {distance})"
    return distance

def evaluate_relations(objects: List[DetectedObject], distance_equality_threshold: float) -> List[List[Relations]]:
    dim = len(objects)
    ssr_matrix = [[Relations() for _ in range(dim)] for _ in range(dim)]

    evaluate_contact_relations(objects, ssr_matrix)
    evaluate_static_relations(objects, ssr_matrix)
    evaluate_dynamic_relations(objects, ssr_matrix, distance_equality_threshold)

    return ssr_matrix

def evaluate_contact_relations(objects: List[DetectedObject], ssr_matrix: List[List[Relations]]):
    dim = len(objects)
    for subject_index in range(dim):
        for object_index in range(subject_index + 1, dim):
            subject_bb = objects[subject_index].bounding_box
            object_bb = objects[object_index].bounding_box

            if is_colliding(subject_bb, object_bb):
                ssr_matrix[subject_index][object_index].contact = True
                ssr_matrix[object_index][subject_index].contact = True

def evaluate_static_relations(objects: List[DetectedObject], ssr_matrix: List[List[Relations]]):
    dim = len(objects)
    for subject_index in range(dim):
        for object_index in range(subject_index + 1, dim):
            subject = objects[subject_index]
            object = objects[object_index]
            subject_bb = subject.bounding_box
            object_bb = object.bounding_box

            if subject_bb.x1 < object_bb.x0:
                ssr_matrix[subject_index][object_index].static_left_of = True
                ssr_matrix[object_index][subject_index].static_right_of = True
            elif subject_bb.x0 > object_bb.x1:
                ssr_matrix[subject_index][object_index].static_right_of = True
                ssr_matrix[object_index][subject_index].static_left_of = True

            if subject_bb.y1 < object_bb.y0:
                ssr_matrix[subject_index][object_index].static_below = True
                ssr_matrix[object_index][subject_index].static_above = True
            elif subject_bb.y0 > object_bb.y1:
                ssr_matrix[subject_index][object_index].static_above = True
                ssr_matrix[object_index][subject_index].static_below = True

            if subject_bb.z1 < object_bb.z0:
                ssr_matrix[subject_index][object_index].static_behind_of = True
                ssr_matrix[object_index][subject_index].static_in_front_of = True
            elif subject_bb.z0 > object_bb.z1:
                ssr_matrix[subject_index][object_index].static_in_front_of = True
                ssr_matrix[object_index][subject_index].static_behind_of = True

            if (object_bb.x0 < subject_bb.x0 and subject_bb.x1 < object_bb.x1 and
                object_bb.z0 < subject_bb.z0 and subject_bb.z1 < object_bb.z1 and
                object_bb.y0 < subject_bb.y0 and subject_bb.y0 <= object_bb.y1):
                ssr_matrix[subject_index][object_index].static_inside = True
                ssr_matrix[object_index][subject_index].static_surround = True
            elif (object_bb.x0 > subject_bb.x0 and subject_bb.x1 > object_bb.x1 and
                  object_bb.z0 > subject_bb.z0 and subject_bb.z1 > object_bb.z1 and
                  object_bb.y0 > subject_bb.y1 and subject_bb.y1 >= object_bb.y1):
                ssr_matrix[subject_index][object_index].static_surround = True
                ssr_matrix[object_index][subject_index].static_inside = True

def evaluate_dynamic_relations(objects: List[DetectedObject], ssr_matrix: List[List[Relations]], distance_equality_threshold: float):
    dim = len(objects)
    for i in range(dim):
        for j in range(i + 1, dim):
            object_a = objects[i]
            object_b = objects[j]
            object_a_bb = object_a.bounding_box
            object_b_bb = object_b.bounding_box
            object_a_past_bb = object_a.past_bounding_box
            object_b_past_bb = object_b.past_bounding_box

            delta_ab = distance_between(object_a_bb, object_b_bb)
            delta_ab_past = distance_between(object_a_past_bb, object_b_past_bb)

            p1 = is_colliding(object_a_bb, object_b_bb) and is_colliding(object_a_past_bb, object_b_past_bb)
            p2 = not is_colliding(object_a_bb, object_b_bb) and not is_colliding(object_a_past_bb, object_b_past_bb)

            if p1:
                p3 = distance_between(object_a_bb, object_a_past_bb) < (distance_equality_threshold / 2)
                p4 = distance_between(object_b_bb, object_b_past_bb) < (distance_equality_threshold / 2)

                if p3 and p4:
                    ssr_matrix[i][j].dynamic_moving_together = True
                    ssr_matrix[j][i].dynamic_moving_together = True
                elif not p3 and not p4:
                    ssr_matrix[i][j].dynamic_halting_together = True
                    ssr_matrix[j][i].dynamic_halting_together = True
                elif p3 ^ p4:
                    ssr_matrix[i][j].dynamic_fixed_moving_together = True
                    ssr_matrix[j][i].dynamic_fixed_moving_together = True
            elif p2:
                if delta_ab - delta_ab_past < -distance_equality_threshold:
                    ssr_matrix[i][j].dynamic_getting_close = True
                    ssr_matrix[j][i].dynamic_getting_close = True
                elif delta_ab - delta_ab_past > distance_equality_threshold:
                    ssr_matrix[i][j].dynamic_moving_apart = True
                    ssr_matrix[j][i].dynamic_moving_apart = True
                else:
                    ssr_matrix[i][j].dynamic_stable = True
                    ssr_matrix[j][i].dynamic_stable = True

def read_json_file(file_path: str) -> List[DetectedObject]:
    # Load the JSON data from the file
    with open(file_path, 'r') as file:
        data = json.load(file)
    # Initialize an empty list to store DetectedObject instances
    detected_objects = []
    # Iterate over each entry in the JSON data
    for entry in data:
        # Extract bounding box information
        bounding_box_data = entry["bounding_box"]
        past_bounding_box_data = entry["past_bounding_box"]
        # Create BoundingBox instances for the current and past bounding boxes
        bounding_box = BoundingBox(**bounding_box_data)
        past_bounding_box = BoundingBox(**past_bounding_box_data)
        # Create a DetectedObject instance
        detected_object = DetectedObject(
            class_name=entry["class_name"],
            bounding_box=bounding_box,
            past_bounding_box=past_bounding_box
        )
        # Append the DetectedObject instance to the list
        detected_objects.append(detected_object)
    return detected_objects

def ssr_matrix_to_json(ssr_matrix: List[List[Relations]]) -> List[Dict[str, Any]]:
    json_output = []
    
    for subject_index, row in enumerate(ssr_matrix):
        for object_index, relations in enumerate(row):
            if subject_index != object_index:  # Skip relations with itself
                # Static relations
                if relations.static_left_of:
                    json_output.append({
                        "subject_index": subject_index,
                        "relation_name": "left of",
                        "object_index": object_index
                    })
                if relations.static_right_of:
                    json_output.append({
                        "subject_index": subject_index,
                        "relation_name": "right of",
                        "object_index": object_index
                    })
                if relations.static_above:
                    json_output.append({
                        "subject_index": subject_index,
                        "relation_name": "above",
                        "object_index": object_index
                    })
                if relations.static_below:
                    json_output.append({
                        "subject_index": subject_index,
                        "relation_name": "below",
                        "object_index": object_index
                    })
                if relations.static_in_front_of:
                    json_output.append({
                        "subject_index": subject_index,
                        "relation_name": "in front of",
                        "object_index": object_index
                    })
                if relations.static_behind_of:
                    json_output.append({
                        "subject_index": subject_index,
                        "relation_name": "behind of",
                        "object_index": object_index
                    })
                
                # Dynamic relations
                if relations.dynamic_stable:
                    json_output.append({
                        "subject_index": subject_index,
                        "relation_name": "stable",
                        "object_index": object_index
                    })
                if relations.dynamic_moving_together:
                    json_output.append({
                        "subject_index": subject_index,
                        "relation_name": "moving together",
                        "object_index": object_index
                    })
                if relations.dynamic_halting_together:
                    json_output.append({
                        "subject_index": subject_index,
                        "relation_name": "halting together",
                        "object_index": object_index
                    })
                if relations.dynamic_fixed_moving_together:
                    json_output.append({
                        "subject_index": subject_index,
                        "relation_name": "fixed moving together",
                        "object_index": object_index
                    })
                if relations.dynamic_getting_close:
                    json_output.append({
                        "subject_index": subject_index,
                        "relation_name": "getting close",
                        "object_index": object_index
                    })
                if relations.dynamic_moving_apart:
                    json_output.append({
                        "subject_index": subject_index,
                        "relation_name": "moving apart",
                        "object_index": object_index
                    })
    
    return json_output

def main():
    # Read the input JSON file
    input_data = read_json_file(input_file)

    # Process the relationships
    processed_data = evaluate_relations(input_data, distance_equality_threshold=0.5)
    json_data = ssr_matrix_to_json(processed_data)

    # Write the json data to the output JSON file
    with open(output_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    #print(f"Processed data has been written to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate spatial relations between objects and save to json files')
    parser.add_argument('-input_folder',type=str, required=True, help='input file path')
    parser.add_argument('-output_folder',type=str, required=True, help='output file path')
    args = parser.parse_args()

    take_nums = os.listdir(args.input_folder)
    for take_num in take_nums: # iterate for each take
        input_file_path = os.path.join(args.input_folder, take_num, "3d_objects")
        files = os.listdir(input_file_path)
        for file in files: # iterate for each frame
            input_file = os.path.join(input_file_path, file)
            output_path = os.path.join(args.output_folder, take_num, "spatial_relations")
            os.makedirs(output_path, exist_ok=True)
            output_file = os.path.join(output_path, file)
            main()
        print(f"All data from take: {take_num} saved to: {output_path}")

    print("All spatial relation data saved to: ", args.output_folder)
