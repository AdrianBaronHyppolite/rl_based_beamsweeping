import numpy as np 

class AntennaModel:
    def __init__(self, fc_ghz):
        self.fc_hz = fc_ghz * 1e9
        self.gain_max = 8 # in (dBi).
        # Other antenna parameters. 
        self._AEmax = 30 # Element pattern (dB).
        self._SLAv = 30 # Vertical pattern (dB).
        self._theta3dB = 65 # Vertical pattern (degrees).
        self._AEhmax = 30 # Horizontal pattern (dB).
        self._phi3dB = 65 # Horizontal pattern (degrees).
        return 

    def calc_ant_gain(self, theta, phi, theta_s, phi_s):
        """This function calculates the antenna gain in dBi of a directional
        antenna given the reference angles:
            - theta_s: The vertical steering angle;
            - phi_s: The horizontal steering angle;
            - theta: The vertical reference angle;
            - phi: The horizontal reference angle;
        """
        A = self._calc_ant_pattern(theta, phi, theta_s, phi_s)
        F = self._calc_field_gain(A)
        return F

    def calc_ant_gain_aligned(self):
        return self.calc_ant_gain(theta=90, phi=0)

    def calc_ant_gain_random(self):
        theta = np.random.randint(0, 180)
        phi = np.random.randint(-180, 180)
        return self.calc_ant_gain(theta, phi)

    def _calc_field_gain(self, A_dB):
        """Assumption: Perfect polarization."""
        return A_dB

    def _calc_ant_pattern(self, theta, phi, theta_s, phi_s):
        AE = self._calc_element_pattern(theta, phi)
        AF = self._calc_array_factor(theta, phi, theta_s, phi_s)
        A = AE + AF
        return A

    def _calc_element_pattern(self, theta, phi):
        AEv = self._calc_vertical_pattern(theta)
        AEh = self._calc_horizontal_pattern(phi)
        AE = self.gain_max - np.min([-(AEv + AEh), self._AEmax])
        return AE

    def _calc_array_factor(self, theta, phi, theta_s, phi_s, rho=1):
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)
        theta_s = np.deg2rad(theta_s)
        phi_s = np.deg2rad(phi_s)
        
        # Wavelength in meters.
        def wl(f): return 3e8/f
        # No. of antenna elements.
        def ne(f):
            if f==700e6:
                return 2*2
            if f==4e9:
                return 4*4
            if f==30e9:
                return 8*8
            assert True==False, "[Error] Invalid frequency!"
            return

        # Vertical spacing between antenna elements (in meters).
        dv = lambda f: wl(f)/2
        # Horizontal spacing between antenna elements (in meters).
        dh = lambda f: wl(f)/2

        n = ne(self.fc_hz)
        a = np.ones(n)/np.sqrt(n)
        m = int(np.sqrt(n))
        w = np.ones((m, m), dtype="complex_")

        for p in range(m):
            for r in range(m):
                psi_p = np.cos(theta) - np.cos(theta_s)
                psi_r = np.sin(theta)*np.sin(phi)-np.sin(theta_s)*np.sin(phi_s)
                term_p = p*dv(self.fc_hz)*psi_p/wl(self.fc_hz)
                term_r = r*dh(self.fc_hz)*psi_r/wl(self.fc_hz)
                w[p,r] = np.exp(2j*np.pi*(term_p + term_r))

        w = w.flatten()
        aw = np.power(np.absolute(np.dot(a, w.T)), 2)
        if rho*(aw - 1) == -1:
            AF = -100
        else:
            AF = 10*np.log10(1 + rho*(aw - 1))
        return AF

    def _calc_vertical_pattern(self, theta):
        AEv = 12*np.power((theta-90)/self._theta3dB, 2)
        AEv = -np.min([AEv, self._SLAv])
        return AEv

    def _calc_horizontal_pattern(self, phi):
        AEh = 12*np.power(phi/self._phi3dB, 2)
        AEh = -np.min([AEh, self._AEhmax])
        return AEh
    
    def get_ang(self, loc):
        return np.angle(loc[0] + 1j*loc[1], deg=True)
    
    def gen_codebook(self, nh_beams, nv_beams):
        thetas = np.linspace(-35, 35, nv_beams) if nv_beams > 1 else [0]
        phis = np.linspace(-45, 45, nh_beams) if nh_beams > 1 else [0]
        id = 0
        codebook = {}
        for phi in phis:
            for theta in thetas:
                codebook[id] = (phi, theta+90)
                id += 1
        return codebook
    

