import os
import glob
import csv

from auto_tagging_engine import AutoTagEngine


# Get paths of test images
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

image_paths = []

exts = ["*.JPG", "*.jpg"]

for ext in exts:
	image_path_pattern = BASE_DIR + "/test_images/" + ext
	img_paths = glob.glob(image_path_pattern)
	image_paths.extend(img_paths)


print('Process ', len(image_paths), ' photos')

# Do tagging
results = AutoTagEngine.do_tagging_process(image_paths)

# Save result to file
outfile_name = 'auto_tag_result.csv'
outfile_path = BASE_DIR + "/" + outfile_name


with open(outfile_path, mode='w') as result_file:
    file_writer = csv.writer(result_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    columns = ['image_name']
    columns.extend(AutoTagEngine.label_5000_list)

    file_writer.writerow(columns)
    for img_path in results:

        parent_path, file_name = os.path.split(img_path)
        predict_list_5000 = results[img_path]

        row = []
        row.append(file_name)
        row.extend(predict_list_5000)

        file_writer.writerow(row)
