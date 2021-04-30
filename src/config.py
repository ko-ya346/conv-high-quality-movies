class CFG:
    DIR = '.'    
    shrink_scale = 16
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
    trained_param = f'{DIR}/output/models/4-3/gen_199.pytorch'
    IMG_DIR = f'{DIR}/input/4-3/images/high_2'
    
    OUTPUT_IMG = f'{DIR}/output/images/4-3/high_2'
    OUTPUT_MODEL = f'{DIR}/output/models/4-3_mse'
