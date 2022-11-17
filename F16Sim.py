from nlplant import nlplant_


NrStates = 18
xu = [0.0] * NrStates
xdot = [0.0] * NrStates

feet_to_m = 0.3048
lbf_to_N = 4.448222
lb_ft2_to_Pa = 47.880258888889

xu_IU_to_SI = [1.0] * NrStates
xdot_IU_to_SI = [1.0] * NrStates

def Init_xu():
    xu[0]  = 0.0
    xu[1]  = 0.0
    xu[2]  = 0.0
    xu[3]  = 0.0
    xu[4]  = 0.0
    xu[5]  = 0.0
    xu[6]  = 100.0
    xu[7]  = 0.0
    xu[8]  = 0.0
    xu[9]  = 0.0
    xu[10] = 0.0
    xu[11] = 0.0
    xu[12] = 5000.0*lbf_to_N
    xu[13] = 0.0
    xu[14] = 0.0
    xu[15] = 0.0
    xu[16] = 0.0
    xu[17] = 1.0

def Init_xu_IU_to_SI():
    for i in range(NrStates):
        xu_IU_to_SI[i] = 1.0
    for i in range(3):
        xu_IU_to_SI[i] = feet_to_m
    xu_IU_to_SI[6] = feet_to_m
    xu_IU_to_SI[12] = lbf_to_N

def Init_xdot_IU_to_SI():
    for i in range(NrStates):
        xdot_IU_to_SI[i] = 1.0
    for i in range(NrStates):
        xdot_IU_to_SI[i] = xu_IU_to_SI[i]
    xdot_IU_to_SI[16] = lb_ft2_to_Pa
    xdot_IU_to_SI[17] = lb_ft2_to_Pa

def InitState():
    Init_xu()
    Init_xu_IU_to_SI()
    Init_xdot_IU_to_SI()

def Convert_xu_IU_to_SI(xu):
    for i in range(NrStates):
        xu[i]*=xu_IU_to_SI[i]

def Convert_xu_SI_to_IU(xu):
    for i in range(NrStates):
        xu[i]/=xu_IU_to_SI[i]

def Convert_xdot_IU_to_SI(xdot):
	for i in range(NrStates):
		xdot[i]*=xdot_IU_to_SI[i]

def Convert_xdot_SI_to_IU(xdot):
	for i in range(NrStates):
		xdot[i]/=xdot_IU_to_SI[i]

def Convert_IU_to_SI(xu, xdot):
	Convert_xu_IU_to_SI(xu)
	Convert_xdot_IU_to_SI(xdot)

def Convert_SI_to_IU(xu, xdot):
	Convert_xu_SI_to_IU(xu)
	Convert_xdot_SI_to_IU(xdot)

def UpdateSimulation(xu, xdot):
    tmp = nlplant_(xu)
    for i in range(len(xdot)):
        xdot[i] = tmp[i]

def UpdateSimulation_plus(xu, xdot):
    Convert_SI_to_IU(xu,xdot)
    UpdateSimulation(xu,xdot)

def RunSimulation():
    RunNr = 0
    dt = 0.01
    # while True:
    for i in range(1000):
        print('step:', i)
        UpdateSimulation_plus(xu,xdot)
        for i in range(NrStates-1):
            xu[i]+=xdot[i]*dt

def main(argv):
    InitState()
    assert len(argv) <= NrStates
    for i,v in enumerate(argv):
        xu[i] = v
    if 0:
        RunSimulation()
    else:
        UpdateSimulation_plus(xu, xdot)
    

if __name__ == '__main__':
    parameters = " 14.3842921301 0.0 999.240528869 0.0 0.0680626236787 0.0 100.08096494 0.121545455798 0.0 0.0 -0.031583522788 0.0 20000.0 0.0 0.0 0.0 0.0 1.0"
    data = list(map(float, parameters.strip().split(' ')))
    main(data)
    print(data)
    print(xdot)
