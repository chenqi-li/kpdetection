import nni
import numpy as np

params = {
    'features': 512,
    'lr': 0.001,
    'momentum': 0,
}



# searchSpace:
#     reisin_saturation:
#       _type: choice
#       _value: [0.3, 0.6, 0.9],
#     reisin_value:
#       _type: choice
#       _value: [0.0, 0.2, 0.4, 0.6]
#     rendering_mode:
#       _type: choice
#       _value: ['alpha', 'premultiply', 'additive', 'multiply']
    
optimized_params = nni.get_next_parameter()
params.update(optimized_params)

result = np.random.rand(1,1)
nni.report_final_result(result)
