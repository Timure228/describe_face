import pandas as pd

with open("data/list_attr_celeba.txt", "r") as f:
    file = f.read()
    arr = file.split()
    attributes = arr[1:41]
    # print(arr[42:82])

    image_attributes = []
    for i in range(1, 15001): # up to 15'000
        next_img_id = (i*40) + i
        image_attributes.append(arr[next_img_id+1: next_img_id+41])

image_attributes = [['0' if s == '-1' else s for s in x] for x in image_attributes]
print(image_attributes)
dataframe = pd.DataFrame(data=image_attributes, columns=attributes)
dataframe.to_csv("C:/Users/tymur.arduch/PycharmProjects/describe_face/data/celeba.csv", index_label="ID")
