import glob
import cv2
from torchvision.io import read_image
from torchvision.transforms.functional import crop
from torchvision.transforms import Resize
from torchvision.utils import save_image

# Get list of people
people_list = glob.glob("dataset/lfw_funneled/*")

# Get the peoples images dirs
face_list = {}

for people in people_list:
    face_list[people] = glob.glob(people + "/*")
    
# Filter by people with 2 or more images
valid_people = {}

for people in list(face_list.keys()):
    if len(face_list[people]) > 1:
        valid_people[people] = face_list[people]
        
len(face_list.keys()), len(valid_people.keys())

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

resize_transform = Resize(128)
i = 0
n_errs = 0
total = len(list(valid_people.keys()))

new_dirs = {}

for person in valid_people:
    person_faces = valid_people[person]
    new_persons_dirs = []
    i += 1
    #print(i, total)
    for img_dir in person_faces:
        img = read_image(img_dir)
        face_coords = face_classifier.detectMultiScale(
            img[0].numpy(), scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )
        if len(face_coords) > 0:
            cropped_img = crop(img, face_coords[0][0], face_coords[0][1], face_coords[0][2], face_coords[0][3])
            cropped_img = resize_transform(cropped_img)
            save_image(cropped_img.float() / 255.0, img_dir.split(".")[0] + "_cropped.jpg")
        else:
            n_errs += 1
            
print(n_errs)