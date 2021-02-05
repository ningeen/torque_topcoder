import os
import sys
import shutil

if len(sys.argv) != 4:
    TRAIN_AUDIO="../../training/"
    TRAIN_GT="../../"
    OUTPUT_DIR="./"
else:
    TRAIN_AUDIO=sys.argv[1]
    TRAIN_GT=sys.argv[2]
    OUTPUT_DIR=sys.argv[3]

src = '/code/weights/'
src_files = os.listdir(src)
for file_name in src_files:
    full_file_name = os.path.join(src, file_name)
    if os.path.isfile(full_file_name):
        shutil.copy(full_file_name, OUTPUT_DIR)