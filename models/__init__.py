import warnings

# CPF
try:
    import manopth
except ImportError:
    warnings.warn('CPF needs lib "manopth", You can install it according to https://github.com/lixiny/manopth/tree/0fe3850db7973ba0f3dd11235fea6c9fb0c55389.')

try:
    import liegroups
except ImportError:
    warnings.warn('CPF needs lib "liegroups".You can install it according to https://github.com/utiasSTARS/liegroups')
try:
    from .CPF import *
except Exception as e:
    print("CPF: ", e)


# Garmentnets
try:
    import igl
except ImportError:
    warnings.warn('GarmentNet needs lib "igl". You can install it according to https://github.com/libigl/libigl-python-bindings.')

try:
    import torch_scatter
except:
    warnings.warn('Garmentnets needs lib "torch_scatter"')

try:
    import zarr
except:
    warnings.warn('Garmentnets needs lib zarr')

try:
    from .Garmentnets import *
except Exception as e:
    print("Garmentnets: ", e)

try:
    from .ASDF import *
except Exception as e:
    print(e)

# CAPTRA
try:
    from .CAPTRA import *
except Exception as e:
    print(e)