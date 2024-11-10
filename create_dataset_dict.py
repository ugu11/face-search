import glob
import json

# Get list of people
people_list = glob.glob("dataset/lfw_funneled/*")

# Get the peoples images dirs
face_list = {}

for people in people_list:
    face_list[people] = glob.glob(people + "/*")
        
# Read cropped image dirs
face_list_2 = {}

for people in people_list:
    face_list_2[people] = glob.glob(people + "/*_cropped.jpg")
    
# Filter by valid people
valid_people2 = {}

for people in list(face_list_2.keys()):
    if len(face_list_2[people]) > 1:
        valid_people2[people] = face_list_2[people]
        

with open("face_dataset.json", "w") as f: 
    json.dump(valid_people2, f)