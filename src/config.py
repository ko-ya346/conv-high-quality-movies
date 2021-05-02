import sys

sys.path.append('../')
from settings import INPUT_DIR, OUTPUT_DIR, LINE_TOKEN


class CFG:
        
    shrink_scale = 3 
    ### 16:9
    # xsize = 
    # ysize = 
    ### 4:3
    xsize = 720
    ysize = 960
    low_xsize = xsize // shrink_scale
    low_ysize = ysize // shrink_scale
    batch_size = 2
    epoch = 30
 #   trained_param = f'{OUTPUT_DIR}/models/4-3/gen_199.pytorch'
    trained_param = False
    IMG_DIR = f'{INPUT_DIR}/4-3/images/high_2'
    
    OUTPUT_IMG = f'{OUTPUT_DIR}/images/4-3/high_2'
    OUTPUT_MODEL = f'{OUTPUT_DIR}/models/4-3_mse'
    LINE_TOKEN = LINE_TOKEN
