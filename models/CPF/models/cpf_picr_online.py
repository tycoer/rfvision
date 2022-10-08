from mmdet3d.models.builder import DETECTORS
from .cpf_picr_offline import CPFPicrOffline

@DETECTORS.register_module()
class CPFPicrOnline(CPFPicrOffline):
    pass
if __name__ == '__main__':
    pass