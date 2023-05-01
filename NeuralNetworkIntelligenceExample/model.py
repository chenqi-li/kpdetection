import nni
import numpy as np
import pandas as pd

params = {
    'ReisinSaturation': 0.3,
    'ReisinValue': 0.6,
    'RenderingMode': 0,
}

convert = {0:'Alpha', 1:'Premultiply', 2:'Additive', 3:'Multiply'}

optimized_params = nni.get_next_parameter()
params.update(optimized_params)

print(params)

df = pd.read_csv("D:\Chenqi\KP Detection\\NeuralNetworkIntelligenceExample\gridsearchtable.csv") 
row = df.loc[((df['Reisin Saturation'] == params['ReisinSaturation']) & (df['Reisin Value'] == params['ReisinValue']) & (df['Rendering Mode'] == convert[params['RenderingMode']]))]

result = float(row['Mean'])

# result = np.random.rand(1,1)[0][0]
print(result)
# nni.report_intermediate_result(result)
nni.report_final_result(result)