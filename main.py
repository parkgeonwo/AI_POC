
from Preprocess import Box_extract


box_extract = Box_extract( upload_image_path="/home/matrix/Desktop/code/AI_POC/document/car_document.jpg" )

canny_image = box_extract.canny_image

detect_box_num = 3
box_center_list = [[0, 0.5, 1],
                    [0, 0.1617, 0.2318, 0.3073, 0.3733, 0.442, 0.5054, 0.5701, 1 ],
                    [0, 0.0513, 0.1099, 0.1526, 0.1954, 0.2357, 0.2772, 0.3541, 0.419,1 ],
                    None ]


box_extract.detect_box(canny_image, detect_box_num, box_center_list, show_image = True)










