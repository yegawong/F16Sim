from mexndinterp import interpn
HIFI_GLOBAL_TXT_CONTENT = {}

def safe_read_dat(dat_name) -> list:
    try:
        if dat_name in HIFI_GLOBAL_TXT_CONTENT:
            return HIFI_GLOBAL_TXT_CONTENT.get(dat_name)
        
        path = r'./data/' + dat_name
        with open(path, 'r', encoding='utf-8') as file:
            content = file.read()
            content = content.strip()
            data_str = [value for value in content.split(' ') if value]
            data = list(map(float, data_str))
            HIFI_GLOBAL_TXT_CONTENT[dat_name] = data
            return data
    except OSError:
        print("Cannot find file {} in current directory".format(path))
        return []


def get_table(gain_params, ndinfo, dat_name):
    assert len(gain_params) == len(ndinfo)
    nDimension = len(ndinfo)
    X = [None] * nDimension
    x = [None] * nDimension
    DATA = safe_read_dat(dat_name)

    def fun(x):
        return {0: getALPHA1(), 1: getBETA1(), 2: getDH1()}.get(x, None)
    for i, v in enumerate(gain_params):
        X[i] = fun(i)
    for i, v in enumerate(gain_params):
        x[i] = v
    return interpn(X, DATA, x, ndinfo)


def getALPHA1():
    return safe_read_dat(r'ALPHA1.dat')


def getALPHA2():
    return safe_read_dat(r'ALPHA2.dat')


def getBETA1():
    return safe_read_dat(r'BETA1.dat')


def getDH1():
    return safe_read_dat(r'DH1.dat')


def getDH2():
    return safe_read_dat(r'DH2.dat')


def _Cx(alpha, beta, dele):
    gain = [alpha, beta, dele]
    ndinfo = [20, 19, 5]
    path = r'CX0120_ALPHA1_BETA1_DH1_201.dat'
    return get_table(gain, ndinfo, path)


def _Cz(alpha, beta, dele):
    gain = [alpha, beta, dele]
    ndinfo = [20, 19, 5]
    path = r'CZ0120_ALPHA1_BETA1_DH1_301.dat'
    return get_table(gain, ndinfo, path)


def _Cm(alpha, beta, dele):
    gain = [alpha, beta, dele]
    ndinfo = [20, 19, 5]
    path = r'CM0120_ALPHA1_BETA1_DH1_101.dat'
    return get_table(gain, ndinfo, path)


def _Cy(alpha, beta):
    gain = [alpha, beta]
    ndinfo = [20, 19]
    path = r'CY0320_ALPHA1_BETA1_401.dat'
    return get_table(gain, ndinfo, path)


def _Cn(alpha, beta, dele):
    gain = [alpha, beta, dele]
    ndinfo = [20, 19, 3]
    path = r'CN0120_ALPHA1_BETA1_DH2_501.dat'
    return get_table(gain, ndinfo, path)


def _Cl(alpha, beta, dele):
    gain = [alpha, beta, dele]
    ndinfo = [20, 19, 3]
    path = r'CL0120_ALPHA1_BETA1_DH2_601.dat'
    return get_table(gain, ndinfo, path)


def _Cx_lef(alpha, beta):
    gain = [alpha, beta]
    ndinfo = [14, 19]
    path = r'CX0820_ALPHA2_BETA1_202.dat'
    return get_table(gain, ndinfo, path)


def _Cz_lef(alpha, beta):
    gain = [alpha, beta]
    ndinfo = [14, 19]
    path = r'CZ0820_ALPHA2_BETA1_302.dat'
    return get_table(gain, ndinfo, path)


def _Cm_lef(alpha, beta):
    gain = [alpha, beta]
    ndinfo = [14, 19]
    path = r'CM0820_ALPHA2_BETA1_102.dat'
    return get_table(gain, ndinfo, path)


def _Cy_lef(alpha, beta):
    gain = [alpha, beta]
    ndinfo = [14, 19]
    path = r'CY0820_ALPHA2_BETA1_402.dat'
    return get_table(gain, ndinfo, path)


def _Cn_lef(alpha, beta):
    gain = [alpha, beta]
    ndinfo = [14, 19]
    path = r'CN0820_ALPHA2_BETA1_502.dat'
    return get_table(gain, ndinfo, path)


def _Cl_lef(alpha, beta):
    gain = [alpha, beta]
    ndinfo = [14, 19]
    path = r'CL0820_ALPHA2_BETA1_602.dat'
    return get_table(gain, ndinfo, path)


def _CXq(alpha):
    gain = [alpha]
    ndinfo = [20]
    path = r'CX1120_ALPHA1_204.dat'
    return get_table(gain, ndinfo, path)


def _CZq(alpha):
    gain = [alpha]
    ndinfo = [20]
    path = r'CZ1120_ALPHA1_304.dat'
    return get_table(gain, ndinfo, path)


def _CMq(alpha):
    gain = [alpha]
    ndinfo = [20]
    path = r'CM1120_ALPHA1_104.dat'
    return get_table(gain, ndinfo, path)


def _CYp(alpha):
    gain = [alpha]
    ndinfo = [20]
    path = r'CY1220_ALPHA1_408.dat'
    return get_table(gain, ndinfo, path)


def _CYr(alpha):
    gain = [alpha]
    ndinfo = [20]
    path = r'CY1320_ALPHA1_406.dat'
    return get_table(gain, ndinfo, path)


def _CNr(alpha):
    gain = [alpha]
    ndinfo = [20]
    path = r'CN1320_ALPHA1_506.dat'
    return get_table(gain, ndinfo, path)


def _CNp(alpha):
    gain = [alpha]
    ndinfo = [20]
    path = r'CN1220_ALPHA1_508.dat'
    return get_table(gain, ndinfo, path)


def _CLp(alpha):
    gain = [alpha]
    ndinfo = [20]
    path = r'CL1220_ALPHA1_608.dat'
    return get_table(gain, ndinfo, path)


def _CLr(alpha):
    gain = [alpha]
    ndinfo = [20]
    path = r'CL1320_ALPHA1_606.dat'
    return get_table(gain, ndinfo, path)


def _delta_CXq_lef(alpha):
    gain = [alpha]
    ndinfo = [14]
    path = r'CX1420_ALPHA2_205.dat'
    return get_table(gain, ndinfo, path)


def _delta_CYr_lef(alpha):
    gain = [alpha]
    ndinfo = [14]
    path = r'CY1620_ALPHA2_407.dat'
    return get_table(gain, ndinfo, path)


def _delta_CYp_lef(alpha):
    gain = [alpha]
    ndinfo = [14]
    path = r'CY1520_ALPHA2_409.dat'
    return get_table(gain, ndinfo, path)


def _delta_CZq_lef(alpha):
    gain = [alpha]
    ndinfo = [14]
    path = r'CZ1420_ALPHA2_305.dat'
    return get_table(gain, ndinfo, path)


def _delta_CLr_lef(alpha):
    gain = [alpha]
    ndinfo = [14]
    path = r'CL1620_ALPHA2_607.dat'
    return get_table(gain, ndinfo, path)


def _delta_CLp_lef(alpha):
    gain = [alpha]
    ndinfo = [14]
    path = r'CL1520_ALPHA2_609.dat'
    return get_table(gain, ndinfo, path)


def _delta_CMq_lef(alpha):
    gain = [alpha]
    ndinfo = [14]
    path = r'CM1420_ALPHA2_105.dat'
    return get_table(gain, ndinfo, path)


def _delta_CNr_lef(alpha):
    gain = [alpha]
    ndinfo = [14]
    path = r'CN1620_ALPHA2_507.dat'
    return get_table(gain, ndinfo, path)


def _delta_CNp_lef(alpha):
    gain = [alpha]
    ndinfo = [14]
    path = r'CN1520_ALPHA2_509.dat'
    return get_table(gain, ndinfo, path)


def _Cy_r30(alpha, beta):
    gain = [alpha, beta]
    ndinfo = [20, 19]
    path = r'CY0720_ALPHA1_BETA1_405.dat'
    return get_table(gain, ndinfo, path)


def _Cn_r30(alpha, beta):
    gain = [alpha, beta]
    ndinfo = [20, 19]
    path = r'CN0720_ALPHA1_BETA1_503.dat'
    return get_table(gain, ndinfo, path)


def _Cl_r30(alpha, beta):
    gain = [alpha, beta]
    ndinfo = [20, 19]
    path = r'CL0720_ALPHA1_BETA1_603.dat'
    return get_table(gain, ndinfo, path)


def _Cy_a20(alpha, beta):
    gain = [alpha, beta]
    ndinfo = [20, 19]
    path = r'CY0620_ALPHA1_BETA1_403.dat'
    return get_table(gain, ndinfo, path)


def _Cy_a20_lef(alpha, beta):
    gain = [alpha, beta]
    ndinfo = [14, 19]
    path = r'CY0920_ALPHA2_BETA1_404.dat'
    return get_table(gain, ndinfo, path)


def _Cn_a20(alpha, beta):
    gain = [alpha, beta]
    ndinfo = [20, 19]
    path = r'CN0620_ALPHA1_BETA1_504.dat'
    return get_table(gain, ndinfo, path)


def _Cn_a20_lef(alpha, beta):
    gain = [alpha, beta]
    ndinfo = [14, 19]
    path = r'CN0920_ALPHA2_BETA1_505.dat'
    return get_table(gain, ndinfo, path)


def _Cl_a20(alpha, beta):
    gain = [alpha, beta]
    ndinfo = [20, 19]
    path = r'CL0620_ALPHA1_BETA1_604.dat'
    return get_table(gain, ndinfo, path)


def _Cl_a20_lef(alpha, beta):
    gain = [alpha, beta]
    ndinfo = [14, 19]
    path = r'CL0920_ALPHA2_BETA1_605.dat'
    return get_table(gain, ndinfo, path)


def _delta_CNbeta(alpha):
    gain = [alpha]
    ndinfo = [20]
    path = r'CN9999_ALPHA1_brett.dat'
    return get_table(gain, ndinfo, path)


def _delta_CLbeta(alpha):
    gain = [alpha]
    ndinfo = [20]
    path = r'CL9999_ALPHA1_brett.dat'
    return get_table(gain, ndinfo, path)


def _delta_Cm(alpha):
    gain = [alpha]
    ndinfo = [20]
    path = r'CM9999_ALPHA1_brett.dat'
    return get_table(gain, ndinfo, path)


def _eta_el(el):
    gain = [el]
    ndinfo = [20]
    path = r'ETA_DH1_brett.dat'
    return get_table(gain, ndinfo, path)


def hifi_C(alpha, beta, el):
    return (_Cx(alpha,beta,el),
            _Cz(alpha,beta,el),
            _Cm(alpha,beta,el),
            _Cy(alpha,beta),
            _Cn(alpha,beta,el),
            _Cl(alpha,beta,el),
    )


def hifi_damping(alpha):
    return (_CXq(alpha),
            _CYr(alpha),
            _CYp(alpha),
            _CZq(alpha),
            _CLr(alpha),
            _CLp(alpha),
            _CMq(alpha),
            _CNr(alpha),
            _CNp(alpha),

    )

def hifi_C_lef(alpha, beta):
    return (_Cx_lef(alpha,beta) - _Cx(alpha,beta,0),
            _Cz_lef(alpha,beta) - _Cz(alpha,beta,0),
            _Cm_lef(alpha,beta) - _Cm(alpha,beta,0),
            _Cy_lef(alpha,beta) - _Cy(alpha,beta),
            _Cn_lef(alpha,beta) - _Cn(alpha,beta,0),
            _Cl_lef(alpha,beta) - _Cl(alpha,beta,0),)


def hifi_damping_lef(alpha):
    return (_delta_CXq_lef(alpha),
            _delta_CYr_lef(alpha),
            _delta_CYp_lef(alpha),
            _delta_CZq_lef(alpha),
            _delta_CLr_lef(alpha),
            _delta_CLp_lef(alpha),
            _delta_CMq_lef(alpha),
            _delta_CNr_lef(alpha),
            _delta_CNp_lef(alpha),
    )


def hifi_rudder(alpha, beta):
    return (
        _Cy_r30(alpha,beta) - _Cy(alpha,beta),
        _Cn_r30(alpha,beta) - _Cn(alpha,beta,0),
        _Cl_r30(alpha,beta) - _Cl(alpha,beta,0),
    )


def hifi_ailerons(alpha, beta):
    return (_Cy_a20(alpha,beta) - _Cy(alpha,beta),
            _Cy_a20_lef(alpha,beta) - _Cy_lef(alpha,beta) - (_Cy_a20(alpha,beta) - _Cy(alpha,beta)),
            _Cn_a20(alpha,beta) - _Cn(alpha,beta,0),
            _Cn_a20_lef(alpha,beta) - _Cn_lef(alpha,beta) - (_Cn_a20(alpha,beta) - _Cn(alpha,beta,0)),
            _Cl_a20(alpha,beta) - _Cl(alpha,beta,0),
            _Cl_a20_lef(alpha,beta) - _Cl_lef(alpha,beta) - (_Cl_a20(alpha,beta) - _Cl(alpha,beta,0)),
    )


def hifi_other_coeffs(alpha, el):
    return (_delta_CNbeta(alpha),
            _delta_CLbeta(alpha),
            _delta_Cm(alpha),
            _eta_el(el),
            0,
    )
