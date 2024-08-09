# import Craft class
from craft_text_detector import Craft
import os
import shutil
output_dir = 'outputs/'
out = 'output_4/'
craft = Craft(output_dir=output_dir, crop_type="poly", cuda=False)

# set image path and export folder directory
img  = os.listdir('4')

for i in img:
    print('4/'+i)
    image_path = '4/'+i
    prediction_result = craft.detect_text(image_path)
    x = i.split('.')[0]
    #print('outputs/'+i+'_crops/'+'crop_0.png')
    shutil.copy('outputs/'+x+'_crops/'+'crop_0.png','output_4/'+i)
    #break

    # unload models from ram/gpu
craft.unload_craftnet_model()
craft.unload_refinenet_model()

