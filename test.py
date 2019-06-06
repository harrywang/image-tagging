import json
import os
import glob
from auto_tagging_engine import AutoTagEngine


# Get paths of test images
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_path_pattern = BASE_DIR + "/test_images/" + "*.JPG"
image_paths = glob.glob(image_path_pattern)

# Do tagging
results = AutoTagEngine.do_tagging_process(image_paths)

for image, result in results.items():
    print image + ' is labeled'
    print result

# Save result to file
# outfile_name = 'result.json'
# outfile_path = BASE_DIR + "/" + outfile_name

# with open(outfile_path, 'w') as outfile:
    # json.dump(results, outfile)
