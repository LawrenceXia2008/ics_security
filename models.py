import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
__all__ = ['MCCED','MCCED_wo_recon','MCCED_wo_1stg','MCCED_wo_2stg','MCCED_wo_atten_2stg','CCED']
def MCCED():
    import MCCED
    return MCCED.network 

def MCCED_wo_recon():
    import MCCED_wo_recon
    return MCCED_wo_recon.network

def MCCED_wo_atten_2stg():
    import MCCED_wo_atten_2stg
    return MCCED_wo_atten_2stg.network

def MCCED_wo_1stg():
    import MCCED_wo_1stg
    return MCCED_wo_1stg.network

def MCCED_wo_2stg():
    import MCCED_wo_2stg
    return MCCED_wo_2stg.network

def CCED():
    import CCED
    return CCED.network

