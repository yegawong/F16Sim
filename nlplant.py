import torch

def atmos(alt, vt):
    rho0 = 2.377e-3
    tfac =1 - .703e-5*(alt)
    temp = 519.0*tfac
    if alt >= 35000.0:
        temp=390
    rho=rho0*pow(tfac,4.14)
    mach = (vt)/torch.sqrt(1.4*1716.3*temp)
    qbar = .5*rho*pow(vt,2)
    ps   = 1715.0*rho*temp

    if ps == 0:
        ps = 1715

    return (mach, qbar, ps)

def accels(state, xdot):
    grav = 32.174
    sina = torch.sin(state[7])
    cosa = torch.cos(state[7])
    sinb = torch.sin(state[8])
    cosb = torch.cos(state[8])
    vel_u = state[6]*cosb*cosa
    vel_v = state[6]*sinb
    vel_w = state[6]*cosb*sina
    u_dot = cosb*cosa*xdot[6] - state[6]*sinb*cosa*xdot[8] - state[6]*cosb*sina*xdot[7]
    v_dot = sinb*xdot[6] + state[6]*cosb*xdot[8]
    w_dot = cosb*sina*xdot[6] - state[6]*sinb*sina*xdot[8] + state[6]*cosb*cosa*xdot[7]
    nx_cg = 1.0/grav*(u_dot + state[10]*vel_w - state[11]*vel_v) + torch.sin(state[4])
    ny_cg = 1.0/grav*(v_dot + state[11]*vel_u - state[9]*vel_w) - torch.cos(state[4])*torch.sin(state[3])
    nz_cg = -1.0/grav*(w_dot + state[9]*vel_v - state[10]*vel_u) + torch.cos(state[4])*torch.cos(state[3])
    return (nx_cg, ny_cg, nz_cg)

def nlplant_(xu):
    xdot = [None] * 18
    g    = 32.17 
    m    = 636.94 
    B    = 30.0
    S    = 300.0
    cbar = 11.32 
    xcgr = 0.35
    xcg  = 0.30

    Heng = 0.0
    pi   = torch.pi

    Jy  = 55814.0
    Jxz = 982.0
    Jz  = 63100.0
    Jx  = 9496.0

    r2d = 180.0/pi

    ## /* %%%%%%%%%%%%%%%%%%%
    ##         States
    ##     %%%%%%%%%%%%%%%%%%% */

    npos  = xu[0]
    epos  = xu[1]
    alt   = xu[2]
    phi   = xu[3]
    theta = xu[4]
    psi   = xu[5]

    vt    = xu[6]
    alpha = xu[7]*r2d
    beta  = xu[8]*r2d
    P     = xu[9]
    Q     = xu[10]
    R     = xu[11]

    sa    = torch.sin(xu[7])
    ca    = torch.cos(xu[7])
    sb    = torch.sin(xu[8])
    cb    = torch.cos(xu[8])
    tb    = torch.tan(xu[8])

    st    = torch.sin(theta)
    ct    = torch.cos(theta)
    tt    = torch.tan(theta)
    sphi  = torch.sin(phi)
    cphi  = torch.cos(phi)
    spsi  = torch.sin(psi)
    cpsi  = torch.cos(psi)

    if vt <= 0.01 : vt = 0.01

    ## /* %%%%%%%%%%%%%%%%%%%
    ## Control inputs
    ## %%%%%%%%%%%%%%%%%%% */

    T     = xu[12]
    el    = xu[13]
    ail   = xu[14]
    rud   = xu[15]
    lef   = xu[16]

    fi_flag = xu[17]/1

    dail  = ail/21.5
    drud  = rud/30.0
    dlef  = (1 - lef/25.0)

    ## /* %%%%%%%%%%%%%%%%%%
    ## Atmospheric effects
    ## sets dynamic pressure and mach number
    ## %%%%%%%%%%%%%%%%%% */

    temp = atmos(alt, vt)
    mach = temp[0]
    qbar = temp[1]
    ps   = temp[2]

    ## /*
    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## %%%%%%%%%%%%%%%%Dynamics%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## */

    ## /* %%%%%%%%%%%%%%%%%%
    ##     Navigation Equations
    ##     %%%%%%%%%%%%%%%%%% */

    U = vt*ca*cb
    V = vt*sb
    W = vt*sa*cb

    xdot[0] = U*(ct*cpsi) + V*(sphi*cpsi*st - cphi*spsi) + W*(cphi*st*cpsi + sphi*spsi)
    xdot[1] = U*(ct*spsi) + V*(sphi*spsi*st + cphi*cpsi) + W*(cphi*st*spsi - sphi*cpsi)
    xdot[2] = U*st - V*(sphi*ct) - W*(cphi*ct)
    xdot[3] = P + tt*(Q*sphi + R*cphi)
    xdot[4] = Q*cphi - R*sphi
    xdot[5] = (Q*sphi + R*cphi)/ct

    if fi_flag == 1:
        from hifi_F16_AeroData import hifi_C, hifi_damping, hifi_C_lef, \
            hifi_damping_lef, hifi_rudder, hifi_ailerons, hifi_other_coeffs
        temp = hifi_C(alpha,beta,el)
        Cx = temp[0]
        Cz = temp[1]
        Cm = temp[2]
        Cy = temp[3]
        Cn = temp[4]
        Cl = temp[5]

        temp = hifi_damping(alpha)
        Cxq = temp[0]
        Cyr = temp[1]
        Cyp = temp[2]
        Czq = temp[3]
        Clr = temp[4]
        Clp = temp[5]
        Cmq = temp[6]
        Cnr = temp[7]
        Cnp = temp[8]

        temp = hifi_C_lef(alpha,beta)
        delta_Cx_lef = temp[0]
        delta_Cz_lef = temp[1]
        delta_Cm_lef = temp[2]
        delta_Cy_lef = temp[3]
        delta_Cn_lef = temp[4]
        delta_Cl_lef = temp[5]

        temp = hifi_damping_lef(alpha)
        delta_Cxq_lef = temp[0]
        delta_Cyr_lef = temp[1]
        delta_Cyp_lef = temp[2]
        delta_Czq_lef = temp[3]
        delta_Clr_lef = temp[4]
        delta_Clp_lef = temp[5]
        delta_Cmq_lef = temp[6]
        delta_Cnr_lef = temp[7]
        delta_Cnp_lef = temp[8]

        temp = hifi_rudder(alpha,beta)
        delta_Cy_r30 = temp[0]
        delta_Cn_r30 = temp[1]
        delta_Cl_r30 = temp[2]

        temp = hifi_ailerons(alpha,beta)
        delta_Cy_a20     = temp[0]
        delta_Cy_a20_lef = temp[1]
        delta_Cn_a20     = temp[2]
        delta_Cn_a20_lef = temp[3]
        delta_Cl_a20     = temp[4]
        delta_Cl_a20_lef = temp[5]

        temp = hifi_other_coeffs(alpha,el)
        delta_Cnbeta = temp[0]
        delta_Clbeta = temp[1]
        delta_Cm     = temp[2]
        eta_el       = temp[3]
        delta_Cm_ds  = 0
    elif fi_flag == 0:
        from lofi_F16_AeroData import damping, dmomdcon, clcn, cxcm, cz
        dlef = 0.0;     
        temp = damping(alpha)
        Cxq = temp[0]
        Cyr = temp[1]
        Cyp = temp[2]
        Czq = temp[3]
        Clr = temp[4]
        Clp = temp[5]
        Cmq = temp[6]
        Cnr = temp[7]
        Cnp = temp[8]

        temp = dmomdcon(alpha,beta)
        delta_Cl_a20 = temp[0]
        delta_Cl_r30 = temp[1]
        delta_Cn_a20 = temp[2]
        delta_Cn_r30 = temp[3]

        temp = clcn(alpha,beta)
        Cl = temp[0]
        Cn = temp[1]

        temp = cxcm(alpha,el)
        Cx = temp[0]
        Cm = temp[1]

        Cy = -.02*beta + .021*dail + .086*drud

        temp = cz(alpha,beta,el)
        Cz = temp[0]

        delta_Cx_lef    = 0.0
        delta_Cz_lef    = 0.0
        delta_Cm_lef    = 0.0
        delta_Cy_lef    = 0.0
        delta_Cn_lef    = 0.0
        delta_Cl_lef    = 0.0
        delta_Cxq_lef   = 0.0
        delta_Cyr_lef   = 0.0
        delta_Cyp_lef   = 0.0
        delta_Czq_lef   = 0.0
        delta_Clr_lef   = 0.0
        delta_Clp_lef   = 0.0
        delta_Cmq_lef   = 0.0
        delta_Cnr_lef   = 0.0
        delta_Cnp_lef   = 0.0
        delta_Cy_r30    = 0.0
        delta_Cy_a20    = 0.0
        delta_Cy_a20_lef= 0.0
        delta_Cn_a20_lef= 0.0
        delta_Cl_a20_lef= 0.0
        delta_Cnbeta    = 0.0
        delta_Clbeta    = 0.0
        delta_Cm        = 0.0
        eta_el          = 1.0
        delta_Cm_ds     = 0.0

    ## /* %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## compute Cx_tot, Cz_tot, Cm_tot, Cy_tot, Cn_tot, and Cl_tot
    ## (as on NASA report p37-40)
    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% */

    dXdQ = (cbar/(2*vt))*(Cxq + delta_Cxq_lef*dlef)
    Cx_tot = Cx + delta_Cx_lef*dlef + dXdQ*Q
    dZdQ = (cbar/(2*vt))*(Czq + delta_Cz_lef*dlef)
    Cz_tot = Cz + delta_Cz_lef*dlef + dZdQ*Q
    dMdQ = (cbar/(2*vt))*(Cmq + delta_Cmq_lef*dlef)
    Cm_tot = Cm*eta_el + Cz_tot*(xcgr-xcg) + delta_Cm_lef*dlef + dMdQ*Q + delta_Cm + delta_Cm_ds
    dYdail = delta_Cy_a20 + delta_Cy_a20_lef*dlef
    dYdR = (B/(2*vt))*(Cyr + delta_Cyr_lef*dlef)
    dYdP = (B/(2*vt))*(Cyp + delta_Cyp_lef*dlef)
    Cy_tot = Cy + delta_Cy_lef*dlef + dYdail*dail + delta_Cy_r30*drud + dYdR*R + dYdP*P
    dNdail = delta_Cn_a20 + delta_Cn_a20_lef*dlef
    dNdR = (B/(2*vt))*(Cnr + delta_Cnr_lef*dlef)
    dNdP = (B/(2*vt))*(Cnp + delta_Cnp_lef*dlef)
    Cn_tot = Cn + delta_Cn_lef*dlef - Cy_tot*(xcgr-xcg)*(cbar/B) + dNdail*dail + delta_Cn_r30*drud + dNdR*R + dNdP*P + delta_Cnbeta*beta
    dLdail = delta_Cl_a20 + delta_Cl_a20_lef*dlef
    dLdR = (B/(2*vt))*(Clr + delta_Clr_lef*dlef)
    dLdP = (B/(2*vt))*(Clp + delta_Clp_lef*dlef)
    Cl_tot = Cl + delta_Cl_lef*dlef + dLdail*dail + delta_Cl_r30*drud + dLdR*R + dLdP*P + delta_Clbeta*beta
    Udot = R*V - Q*W - g*st + qbar*S*Cx_tot/m + T/m
    Vdot = P*W - R*U + g*ct*sphi + qbar*S*Cy_tot/m
    Wdot = Q*U - P*V + g*ct*cphi + qbar*S*Cz_tot/m
    xdot[6] = (U*Udot + V*Vdot + W*Wdot)/vt
    xdot[7] = (U*Wdot - W*Udot)/(U*U + W*W)
    xdot[8] = (Vdot*vt - V*xdot[6])/(vt*vt*cb)
    L_tot = Cl_tot*qbar*S*B
    M_tot = Cm_tot*qbar*S*cbar
    N_tot = Cn_tot*qbar*S*B
    denom = Jx*Jz - Jxz*Jxz
    xdot[9] =  (Jz*L_tot + Jxz*N_tot - (Jz*(Jz-Jy)+Jxz*Jxz)*Q*R + Jxz*(Jx-Jy+Jz)*P*Q + Jxz*Q*Heng)/denom
    xdot[10] = (M_tot + (Jz-Jx)*P*R - Jxz*(P*P-R*R) - R*Heng)/Jy
    xdot[11] = (Jx*N_tot + Jxz*L_tot + (Jx*(Jx-Jy)+Jxz*Jxz)*P*Q - Jxz*(Jx-Jy+Jz)*Q*R +  Jx*Q*Heng)/denom
    temp = accels(xu,xdot)

    xdot[12]  = temp[0]
    xdot[13]  = temp[1]
    xdot[14]  = temp[2]
    xdot[15]  = mach
    xdot[16]  = qbar
    xdot[17]  = ps

    return xdot