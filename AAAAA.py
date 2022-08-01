#%%
from os import path
import sys
import os

path_base = 'C:\\Users\\akuznets\\Data\\SPHERE\\test\\'

script = open('main_SPHERE.py')
code = script.read()

add_before = "import sys\npath_test = sys.argv[1]\n"
add_after = "".join([
    "save_data = {\n",
    "    \'F\':   F.item(),\n",
    "    \'dx\':  dx.item(),\n",
    "    \'dy\':  dy.item(),\n",
    "    \'r0\':  r0.item(),\n",
    "    \'n\':   n.item(),\n",
    "    \'bg\':  bg.item(),\n",
    "    \'Jx\':  Jx.item(),\n",
    "    \'Jy\':  Jy.item(),\n",
    "    \'Jxy\': Jxy.item(),\n",
    "    \'SR data\': SR(PSF_0), # data PSF\n",
    "    \'SR fit\':  SR(PSF_1), # fitted PSF\n",
    "    \'FWHM fit\':  FitGauss2D(PSF_1), \n",
    "    \'FWHM data\': FitGauss2D(PSF_0.float()),\n",
    "    \'Img. data\': PSF_0.detach().cpu().numpy()*param,\n",
    "    \'Img. fit\':  PSF_1.detach().cpu().numpy()*param}\n",
    "path_save = \'C:/Users/akuznets/Data/SPHERE/fitted/\'\n",
    "index = path.split(path_test)[-1].split(\'_\')[0]\n",
    "with open(path_save+str(index)+\'.pickle\', \'wb\') as handle:\n",
    "    pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)\n",
    "print(\'Saved at:\', path_save+str(index)+\'.pickle\')\n",
])

code = "".join([add_before, code.replace('#Cut', '\'\'\''), add_after])

#%
#files = os.listdir(path_base)
#file = files[10]
#path_data = path.join(path_base,file)
#sys.argv = [path_data, path_data]
#
#exec(code)

#%
files = os.listdir(path_base)
for file in files:
    path_data = path.join(path_base,file)
    sys.argv = [path_data, path_data]

    try:
        exec(code) 
    except RuntimeError:
        print("Oops! Anyway...")



# %%
