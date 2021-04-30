import os
import sys
from dotenv import load_dotenv

sys.path.append('./')
from src.utils import get_line_token

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path)

line_token_path = os.getenv('LINE_TOKEN_PATH')

class CFG:
    INPUT_DIR = os.getenv('INPUT_DIR')
    OUTPUT_DIR = os.getenv('OUTPUT_DIR')
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
    LINE_TOKEN = get_line_token(line_token_path)
