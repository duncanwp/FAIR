# Aerosol indirect effects global model
# Original FORTRAN code by Steve Ghan and colleagues
# Ghan et al, 2013: A simple model of global aerosol indirect effects
# doi 10.1002/jgrd.50567

# Other references:
# Ghan et al., 2011: 10.1029/2011MS000074
# Abdul-Razzak & Ghan, 2000: 10.1029/1999JD901161
# Khairoutdinov & Kogan, 2000: 10.1175/1520-0493(2000)128<0229:ANCPPI>2.0.CO;2
# Martin 1994: 10.1175/1520-0469(1994)051<1823:TMAPOE>2.0.CO;2
# Lacis & Hansen, 1974: 10.1175/1520-0469(1974)031<0118:APFTAO>2.0.CO;2

# TODO:
# give variables more logical names?
# other options aside from default CAM5
# put everything inside arrays where possible
#  - try and avoid looping over the 10 AOD bins
# make callable from emissions dataset/FAIR
# thermodynamic parameter could be a function of GMST
# Wood solver currently does not work for PD zero emissions
# Get rid of runtime warning in solver
# some error checking of inputs 
# what does pdf of cloud fraction actually mean?
# Are burdens used? If so can we work backwards from burdens to emissions?
# AIE2 switch
# back calculate emissions from burdens
# Unit tests

import numpy as np
from scipy.optimize import root
from ..constants.aerosol import *


def kk2000(ql, ndrop, A=1350, B=2.47, C=-1.79):
    """Khairoutdinov-Kogan autoconversion rate calculation
    
    ql: cloud liquid water content after depletion (kg kg**-1)
    ndrop: CCN concentration (cm**-3)"""
    
    return A * ql**B * ndrop**C


def woodmicro2(Q, h, ndrop, qlad, tau_rep, r_emb=R_EMB):

    memb=4.*RHO_WATER*np.pi*r_emb**3/3.    # kg

    # main solver loop
    Ac = kk2000(Q[0], ndrop)
    # Q is flattened
    Kc = BETA * Q[0] * Q[1]
    rv = 3 * Q[1] / (4 * np.pi * RHO_WATER * Q[2])
    Sq = 2 * Q[1] * (0.012 * rv - 0.2)/h
    SN = 2 * ndrop * (0.007 * rv - 0.1)/h
    Q[0] = -tau_rep * (Ac + Kc) + qlad
    Q[1] = kk2000(Q[0], ndrop)/(2*(0.012*rv-0.2)/h-BETA*Q[1])
    Q[2] = 4*np.pi*RHO_WATER*rv**3/(3*Q[1])
    return Q
    

def cloud(sigh, fcloud, w, naer, rad, hygro, sigma, rcrit, ndmin, l_wood, tau_rep, r_emb, ncld=20,
          sccn=0.001):
    """Calculates cloud albedo which is the returned value
    sccn: supersaturation for partitioning by CCN
    
    ncld: number of cloud pdf steps"""

    zeta = 2. * ATEN / 3. * np.sqrt(ALPHA * w /G_THD)
        # Abdul-Razzak and Ghan, eq 10

    logsigma = np.log(sigma)
    scrit = 2. / np.sqrt(hygro) * (ATEN / (3. * rad))**1.5
    eta = 2. * ((ALPHA * w / G_THD)**1.5) / (GAMMA * naer)
        # Abdul-Razzak and Ghan, eq 11

    f1 = 0.5 * np.exp(2.5 * logsigma**2)        # Abdul-Razzak and Ghan, eq 7
    f2 = 1. + 0.25 * logsigma                   # Abdul-Razzak and Ghan, eq 8
    s  = np.sum((f1 * (zeta/eta)**1.5 + f2 * 
        (scrit**2 / (eta + 3.*zeta))**0.75) / scrit**2)                         
    smax = 1. / np.sqrt(s)                      # Abdul-Razzak and Ghan, eq 6
    c = 8. / (3. * np.sqrt(2.*np.pi) * logsigma)
    ndrop = np.sum(naer / (1. + (scrit/smax)**c)) # Ghan et al 2013, eq 1
    ccn   = np.sum(naer / (1. + (scrit/sccn)**c))
    ndrop = np.max(ndrop, ndmin)

    # These results differ slightly from Ghan (2013), as in their FORTRAN code
    # they use sigma=1.8 for all modes which affects the value of c in the 
    # equations above.

    hmean = -0.25 * sigh * np.sqrt(2.*np.pi) * np.log(1./fcloud -1.)
    hmax  = np.maximum(hmean, 0.) + 3. * sigh
    dh    = hmax/ncld
    h     = np.arange(0.5, ncld) * dh
    probh = dh / (sigh * np.sqrt(2*np.pi)) * np.exp(-0.5 * ((h-hmean)/sigh)**2)

    if l_wood:
        # Wood microphysical scheme [repr. 2]: Ghan et al (2013) eqs 11-18
        # Note because of the root-finding mechansim this is much slower that
        # the adiabatic scheme and users will probably not want to run this
        # in ensemble projections.

        # Some code repetition from woodmicro2 but initial guess needed for
        # iterative scheme; refactoring target
        memb  = 4. * RHO_WATER * np.pi * r_emb **3 / 3.
        qlad  = 0.5 * A_WOOD * h / RHO_AIR
        ql    = 0.4 * qlad
        qr    = ((qlad-ql)/tau_rep - kk2000(ql, ndrop))/(BETA * RHO_AIR * ql)
        ndriz = (RHO_AIR * kk2000(ql, ndrop) * h/(2. * memb * 0.007) * (
            4. * np.pi * RHO_WATER/(3. * RHO_AIR * qr))**(1/3.))
        Q     = np.zeros((ncld, 3))
        for i in range(ncld):
            Q[i,:] = root(woodmicro2, [ql[i], qr[i], ndriz[i]], 
                args=(h[i], ndrop, qlad[i], tau_rep, r_emb))['x']
            # Do we need to track Q[:,1] and Q[:,2]?
            # Don't think they are used in future calcs

        lwp   = RHO_AIR * Q[:,0] * h 
        rvol  = (3. * RHO_AIR * Q[:,0] / (
            4. * np.pi * RHO_WATER * ndrop))**(1/3.)
        reff  = rvol/RVOLTOEFF
    else:
        # adiabatic cloud model [repr. 1]: Ghan et al (2013) eqs 8-10
        rvol = (3. * A_WOOD * h/(4. * np.pi * RHO_WATER * ndrop))**(1/3.)
        reff = rvol/RVOLTOEFF
        def _precip_formation():
            # Limits liquid water path to that calculated from the maximum
            # cloud droplet size rcrit.
            hm  = 4. * np.pi * RHO_WATER * ndrop * (
                rcrit * RVOLTOEFF)**3 / (3.*A_WOOD)
            lwp = 0.5 * A_WOOD * hm*hm + A_WOOD*hm*(h-hm)
            return lwp
        lwp = np.where(reff > rcrit, _precip_formation(), 0.5 * A_WOOD * h*h)
        reff[reff > rcrit] = rcrit
        
    # Convert to cloud albedo: geometric optics
    taucld    = (3. * lwp)/(2. * RHO_WATER * reff)    # Ghan et al (2013) eq 6
    cldalbedo = taucld / (8. + taucld)                # Ghan et al (2013) eq 5
    sumcld    = np.sum(cldalbedo * probh)
    sump      = np.sum(probh)  # should be one - could error check here
    suml      = np.sum(lwp*probh)
    cldalbedo = sumcld / sump

    return cldalbedo

    
def ghan_indirect(
        npdf           = 10,       # bins for aerosol mass PDF
        scaleh         = 3000.,    # aerosol scale height, m
        rho_aer        = 1700.,    # aerosol density, kg m**-3
        lifetime       = 4.,       # aerosol lifetime in days
        sigh           = 200.,     # standard deviation of cloud height, m
        w              = 0.3,      # updraft velocity, m s**-2
        ndmin          = 0,        # minimum droplet number, #/m3
        srfalbedo      = 0.1,      # surface albedo

        # The default aerosol parameters are from CAM5.
        aername        = ['Accumulation', 'Aitken', 'Coarse'],
        naernat        = np.array([250., 155., 1.7]) * 1e6, # number m**-3 PI
        radnat         = np.array([0.071, 0.0149, 0.784]) * 1e-6,   # m PI
        sigma          = np.array([1.8, 1.6, 1.8]),     # geom. stdev of bins
        naerpd         = np.array([386., 195., 1.7]) * 1e6, # number m**-3 PD
        radpd          = np.array([0.067, 0.0162, 0.784]) * 1e-6, # m PD

        # Aerosol burdens per mode: mass mixing ratio (kg aerosol per kg air)
        so4nat         = np.array([2.4e-10, 6.48e-12, 7.87e-12]),
        soanat         = np.array([7.38e-10, 5.22e-13, 0]),
        bcnat          = np.array([2.82e-11, 0, 0]),
        pomnat         = np.array([2.86e-10, 0, 0]),
        dustnat        = np.array([1.37e-9, 0, 2.17e-8]),
        seasaltnat     = np.array([7.49e-10, 2.05e-12, 1.14e-8]),
        so4pd          = np.array([7.92e-10, 1.08e-11, 1.83e-11]),
        soapd          = np.array([8.85e-10, 1.94e-12, 0]),
        bcpd           = np.array([7.53e-11, 0, 0]),
        pompd          = np.array([4.67e-10, 0, 0]),

        # Hygroscopic growth factors (to vectorise?)
        hygroso4       = 0.5,
        hygrosoa       = 0.1 ,
        hygroseasalt   = 1.2,
        hygrobc        = 0.,
        hygropom       = 0.,
        hygrodust      = 0.1,

        # emissions properties
        radp           = np.array([0.05e-6, 1., 1.]), # (m)
            # error returned if zero: TODO: eliminate
        emitso4        = 110.,     # anthro sulfate emissions Tg yr-1
        emitsoa        = 14.,      # secondary OA anthro emissions Tg yr-1
        emitp          = 22.,      # primary anthro BC + POM emissions Tg yr-1
        fnew           = 0.5,      # new particle formation fraction
        distanthro     = 'mode1',
        fracmode       = None,
        sccn           = 0.002, 

        # Second indirect effect
        l_aie2         = True,     # include second aerosol indirect effect
        l_wood         = True,     # Wood microphysics,
        tau_rep        = 3600.,    # cloud water replenishment time (s)
        rcrit          = 10.e-6,   # critical droplet radius (m)
                                   # 50e-6 means no 2nd indirect

        # Cloud properties
        cldloc         = 'uniform',
        cldfrac        = None,
        fcloud         = 0.37,      # low cloud fraction
    ):

    """Calculates global aerosol indirect effect from Ghan's simple model.

    Reference:

    Keywords:
        distanthro:
            'area'    : distribute secondary anthropogenic by natural area
            'number'  : distribute secondary anthropogenic by natural number
            'mode1'   : put all secondary anthropogenic on mode 1 (default)
            'pd'      : use present-day simulation to determine distribution
            'specify' : specify distribution
            'ccn'     : distribute secondary anthropogenic by CCN(0.1%)
        fracmode:
            array of fractions of emissions to put on each mode. Used if
            distantrho is 'specify'. ::Check fracmode sums to 1 and is correct len::
        sccn:
            supersaturation for partitioning by CCN.

        cldloc:
            'uniform' : assume globally averaged low cloud fraction
            'local'   : assume PDF of low cloud fraction
        cldfrac:
            array of cloud fractions corresponding to each bin member of the PDF
    """

    s0        = SOLAR_CONSTANT*0.25  # average for geometry
    fanthro   = 1/float(npdf)
    nmodes    = len(aername)
    
    secnat    = so4nat + soanat
    primnat   = bcnat + pomnat
    secpd     = so4pd + soapd
    primpd    = bcpd + pompd
    seasaltpd = seasaltnat
    dustpd    = dustnat
    
    hygronat = (hygroso4 * so4nat +
                hygrosoa * soanat + 
                hygroseasalt * seasaltnat +
                hygrodust * dustnat)/(
                so4nat + soanat + seasaltnat + dustnat + bcnat + pomnat)
    
    hygro    = (hygroso4 * so4pd + 
                hygrosoa * soapd +
                hygroseasalt * seasaltpd +
                hygrodust * dustpd)/(
                so4pd + soapd + seasaltpd + dustpd + bcpd + pompd)
    
    # There is an error in line 423 of the FORTRAN code which uses natural SOA
    # instead of the PD SOA. The results from this Python implementation will
    # therefore not agree.
    
    # lines 828-835 in FORTRAN file

    # calculate the mass of each mode
    masspi = 4./3. * (np.pi * rho_aer * naernat * radnat**3 * 
        np.exp(4.5 * np.log(sigma)**2))
    masspd = 4./3. * (np.pi * rho_aer * naerpd  * radpd**3  * 
        np.exp(4.5 * np.log(sigma)**2))
    sumnat       = np.sum(masspi)
    sumnatso4    = np.sum(so4nat) * RHO_AIR
    sumanthros   = np.sum(secpd - secnat) * RHO_AIR
    sumanthroso4 = np.sum(so4pd - so4nat) * RHO_AIR
    sumanthrosoa = np.sum(soapd - soanat) * RHO_AIR
    sumanthrop   = np.sum(primpd - primnat) * RHO_AIR

    # Assumptions about how to distribute anthropogenic aerosol
    if distanthro != 'specify':
        fracmode = np.zeros(nmodes)
        if distanthro == 'area':
            areaaer = naernat * radnat**3 * np.exp(2. * np.log(sigma)**2)
            sumarea = np.sum(areaaer)
            fracmode = areaaer/sumarea
        elif distanthro == 'number':
            fracmode = naernat/np.sum(naernat)
        elif distanthro == 'mode1':
            fracmode[0] = 1.
        elif distanthro == 'pd':
            def _distribute_by_pd():
                naerp=3. * RHO_AIR * (primpd - primnat) / (4. * np.pi *
                    rho_aer * radp**3 * np.exp(4.5 * np.log(sigma**2)))
                fnewn = (naerpd - naernat - naerp) * masspd/(
                    naerpd * RHO_AIR * (secpd - secnat))
                return fnewn
            fracmode = np.ones(nmodes)
            fnewn    = np.where(masspd > masspi, _distribute_by_pd(), 0.)
            fnewn    = np.maximum(fnewn,0.0)
            fnewn    = np.minimum(fnewn,1.0)
            fracmode = RHO_AIR * (secpd - secnat) / sumanthros
        elif distanthro=='ccn':
            # duplicated somewhat in cloud(): refactoring target
            logsigma = np.log(sigma)
            scrit    = 2. /np.sqrt(hygronat) * (ATEN / (3. * radnat))**1.5
            c        = 8. / (3. * np.sqrt(2. * np.pi) * logsigma)
            ccn      = naernat/(1 + (scrit/sccn)**c)
            sumccn   = np.sum(ccn)
            scrit    = 2./np.sqrt(hygro) * (ATEN/(3.*radpd))**1.5
            fracmode = ccn/sumccn

    # Tg yr-1 natural emissions - although not used elsewhere in emissions driven mode
#    emitnat    = sumnat * scaleh * DAYS_IN_YEAR * A_EARTH/lifetime * 1e-12    #should be 1e-9 for kg?
#    emitnatso4 = sumnatso4 * scaleh * DAYS_IN_YEAR * A_EARTH/lifetime * 1e-12 #should be 1e-9 for kg?

    # Cloud fraction bins
    if cldloc == 'uniform':
        cldfrac = np.ones(npdf)*fcloud
    elif cldloc == 'local':
        gcldfrac = np.sum(fanthro * cldfrac)
        cldfrac = cldfrac * fcloud/gcldfrac
    else:
        raise ValueError("cldloc should be 'uniform' or 'local'")

    # Efficiency target: eliminate loop
    pinet_local = np.zeros(npdf)
    for i in range(npdf):
        picldalbedo = cloud(sigh, cldfrac[i], w, naernat, radnat,
            hygronat, sigma, rcrit, ndmin, l_wood, tau_rep, R_EMB)

        pitranscld  = 1. - picldalbedo
        pinet_local[i] = (1. - cldfrac[i]) * (1. - srfalbedo) + cldfrac[i] * (
            pitranscld * (1. - srfalbedo)/(1. - srfalbedo * picldalbedo))
        pinet=np.sum(pinet_local*fanthro)

    if distanthro!='pd':
        fnewn   = np.ones(nmodes) * fnew

    # global mean burdens (kg/m2)
    ganthroso4 = emitso4 * lifetime/(DAYS_IN_YEAR * A_EARTH) * 1e9 
    ganthrosoa = emitsoa * lifetime/(DAYS_IN_YEAR * A_EARTH) * 1e9
    ganthros   = sumanthros * scaleh # not sure about this
    ganthrop   = emitp   * lifetime/(DAYS_IN_YEAR * A_EARTH) * 1e9

    anthroso4b = np.zeros(npdf)
    anthrosoab = np.zeros(npdf)
    anthrosb   = np.zeros(npdf)
    anthropb   = np.zeros(npdf)
    
    if npdf==1:
        anthroso4b[0] = ganthroso4
        anthrosoab[0] = ganthrosoa
        anthrosb[0]   = ganthros
        anthropb[0]   = ganthrop
    else:
        # efficiency target: exponential pdf here
        anthroso4b[0] = -ganthroso4*np.log(1.-0.5*fanthro)
        anthrosoab[0] = -ganthrosoa*np.log(1.-0.5*fanthro)
        anthrosb[0]   = -ganthros*np.log(1.-0.5*fanthro)
        anthropb[0]   = -ganthrop*np.log(1.-0.5*fanthro)
        for i in range(1,npdf):
            # efficiency target: eliminate loop. Do we really define PDF
            # iteratively based on previous term?
            if np.abs(ganthroso4) > 1e-21:   # not sure why here
                anthroso4b[i] = -ganthroso4 * np.log(
                    np.exp(-anthroso4b[i-1] / ganthroso4) - fanthro)
            else:
                anthroso4b[i] = 0

            if ganthrosoa > 0:
                anthrosoab[i] = -ganthrosoa * np.log(
                    np.exp(-anthrosoab[i-1] / ganthrosoa) - fanthro)
            else:
                anthrosoab[i] = 0.

            if ganthros > 0:
                anthrosb[i] = -ganthros * np.log(
                    np.exp(-anthrosb[i-1] / ganthros) - fanthro)
            else:
                anthrosb[i] = 0.

            if ganthrop > 0:
                anthropb[i] = -ganthrop * np.log(
                    np.exp(-anthropb[i-1] / ganthrop) - fanthro)
            else:
                anthropb[i]=0.
    
    sumso4 = np.sum(anthroso4b) * fanthro
    sumsoa = np.sum(anthrosoab) * fanthro
    
    natconc = rho_aer * 4./3. * np.pi * naernat * radnat**3 * np.exp(
        4.5 * np.log(sigma)**2) # kg / m3
    natemit = natconc * scaleh * (DAYS_IN_YEAR * A_EARTH) / lifetime*1e-9
    
    rad = np.zeros(nmodes)
    naer = np.zeros(nmodes)
    net_local = np.zeros(npdf)
    for i in range(npdf):
        # efficiency target: eliminate loop
        anthroso4conc = anthroso4b[i] / scaleh * fracmode
        anthrosoaconc = anthrosoab[i] / scaleh * fracmode
        anthropconc   = anthropb[i]   / scaleh * fracmode
        anthroconc    = anthroso4conc + anthrosoaconc
        nanthrop      = 3.*anthropconc/(4.
            * np.pi * rho_aer * radp**3) * np.exp(-4.5 * np.log(sigma)**2)
        rad= (3. * (natconc + anthropconc + (1. - fnewn) * anthroconc)/(4.*(naernat+nanthrop)*np.pi*rho_aer))**(1/3.)*np.exp(-1.5*np.log(sigma)**2)
        naer=(natconc+anthropconc+anthroconc)*(naernat+nanthrop)/(natconc+anthropconc+(1-fnewn)*anthroconc)
        hygro=(hygronat*natconc+anthroso4conc*hygroso4+anthrosoaconc*hygrosoa)/(natconc+anthroconc+anthropconc)
        pdcldalbedo = cloud(sigh, cldfrac[i], w, naer, rad, hygro, sigma, rcrit, ndmin, l_wood, tau_rep, R_EMB)
        pdtranscld=1-pdcldalbedo
        net_local[i]=(1-cldfrac[i])*(1-srfalbedo)+cldfrac[i]*pdtranscld*(1-srfalbedo)/(1-srfalbedo*pdcldalbedo)
    pdnet=np.sum(fanthro*net_local)
    
    sabs     = s0 * pdnet
    pdalbedo = 1. - pdnet
    
    return s0*(pdnet-pinet)
