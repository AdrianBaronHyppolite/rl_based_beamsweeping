import numpy as np 
from toolbox import listOfTuples

"""
Antenna Array Model
Authors: Jacek Kibilda
"""

def positionAntennaElements(nx,ny,nz,dx,dy,dz):
    nx_vec = np.arange(nx)
    ny_vec = np.arange(ny)
    nz_vec = np.arange(nz)

    s = (nx*ny*nz,3)
    element_array = np.zeros(s)

    #xPosition
    xPos = dx * nx_vec
    element_array[:,0] = np.tile(xPos, ny * nz)

    #yPosition
    yPos = dy * ny_vec
    element_array[:,1] = np.tile(np.tile(yPos, (nx,1)).T.reshape((nx * ny,1)),(nz,1)).T

    #zPosition
    zPos = dz * nz_vec
    element_array[:,2] = np.tile(zPos, (nx * ny,1)).T.reshape((nx * ny * nz,1)).T

    return element_array


class Omnidirectional:
    def __init__(self, lamb, nx, ny, nz, dx, dy, dz):
        self.lamb = lamb 
        self.nx = nx 
        self.ny = ny 
        self.nz = nz
        self.dx = dx 
        self.dy = dy 
        self.dz = dz 
        return 

    def calc_array_factor(self, theta, phi):
        n = self.nx * self.ny * self.nz 
        return np.ones(n)

#This codebook is generated for the array in y-z plane
#The codebook generates a beam on a grid that spans vAngMin and vAngMax (vertical angles), and hAngMin and hAngMax (horizontal angles); vBeams denotes the number of vertical beams, and hBeams the number of horizontal beams
#Default values correspond to the MHU values
def codebook_generator(wavelen,nx,ny,nz,dx,dy,dz,element_array,hBeams=9,vBeams=7,hAngMin=-45,hAngMax=45,vAngMin=-35,vAngMax=35):

    k = 2 * np.pi / wavelen

    # Defining the RF beamforming codebook in the x-direction
    codebook_size_x = nx
    codebook_size_y = ny
    codebook_size_z = nz

    #NOTE: theta and phi are swapped in definition here to follow the convention used by the MHU
    
    # MHU - np.linspace(-45,45,9) Azimuth
    phi = np.linspace(hAngMin,hAngMax,hBeams)
    
    # MHU - np.linspace(-35,35,7) Elevation; boresight is (0,0) so we need to translate to our domain where boresight is (90,0)
    theta = np.linspace(vAngMin,vAngMax,vBeams) + 90
    
    num_beams = hBeams + vBeams
    
    #Prepare vectors that allocate an angle to each antenna element
    theta_tile = np.tile(theta, (hBeams,1)).T.flatten()
    thetaRad = np.deg2rad(theta_tile)
    phi_tile = np.tile(phi, vBeams)
    phiRad = np.deg2rad(phi_tile)
    
    beam_angles = listOfTuples(theta_tile,phi_tile)
    
    '''
    NB I'm supressing this part of the code to keep compatibility with our previous implementation.
    This way, the steering vector is given by the calc_array_factor() function. 
    Here, we only compute the dictionary of angles.

    Notice that both implementations yield the same results. To see that, 
    uncomment code blocks (J.1 and J.2).
    '''
    phase_shift = None
    # # J.1: Uncomment to activate Jacek's implementation. {{{
    # x_component = k * element_array[:,0] * np.multiply(np.cos(phiRad),np.sin(thetaRad))[:,None] #Currently not used
    # y_component = k * element_array[:,1] * np.multiply(np.sin(phiRad),np.sin(thetaRad))[:,None]
    # z_component = k * element_array[:,2] * np.cos(thetaRad)[:,None]
    # phase_shift = np.exp(-1j * y_component - 1j * z_component)
    # # }}}

    #Indices in the range 1:63; the index of antenna beam at boresight is 32
    all_beams = np.arange(num_beams) + 1
    
    return phase_shift, all_beams, beam_angles

class AntennaArray:
    def __init__(self, wavelen, nx, ny, nz, dx, dy, dz, hbeams=0, vbeams=0,
            hangmin=0, hangmax=0, vangmin=0, vangmax=0, elem_type='isotropic'):
        self.wavelen = wavelen
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        
        #Antenna Element type; default is 'isotropic'; other models are currently not supported
        self.elem_type = elem_type
        
        self.element_array = positionAntennaElements(nx, ny, nz, dx, dy, dz)
        
        if hbeams * vbeams > 0:
            self.codebook, self.beam_ids, self.beam_angles = \
                codebook_generator(wavelen, nx, ny, nz, dx, dy, dz, 
                self.element_array, hbeams, vbeams, hangmin, hangmax, vangmin, 
                vangmax)
            self.codebook_ids = np.arange(len(self.beam_angles))
        else:
            self.codebook = self.beam_ids = self.beam_angles = self.codebook_ids = None

        return

    def calc_array_factor(self, theta, phi):
        thetaRad = theta#np.deg2rad(theta)
        phiRad = phi#np.deg2rad(phi)
        
        k = 2 * np.pi / self.wavelen
        
        n = self.nx * self.ny * self.nz
        
        a = np.ones(n)/np.sqrt(n) #normalize the output
        
        x_component = k * self.element_array[:,0] * np.cos(phiRad) * np.sin(thetaRad)
        y_component = k * self.element_array[:,1] * np.sin(phiRad) * np.sin(thetaRad)
        z_component = k * self.element_array[:,2] * np.cos(thetaRad)

        w = np.exp(1j * x_component.T + 1j * y_component.T + 1j * z_component.T)
        
        aw = a * w
  
        return aw


    def steering_vec(self, beam_id, phis=None):
        '''
        Input: 
            - beam_id: The codebook id that will dictated the beam direction;
            - relative_phis: The relative angles (horizontal axis) corresponding
                to the location of receivers.
        Output:
            - aw: The radiation pattern
        Returns:
            - Codebook: Vector with the radition pattern corresponding to beam_id.
            - Beam angles: Humane beam information corresponding to the beam_id
                and thus the radiation vector within the codebook.
        '''
        # Validation of the setup.
        err_msg = "Error! You have not initialized the codebook. " 
        err_msg += "Please, make sure hbeams and vbeams > 0."        
        assert len(self.beam_angles) > 0, err_msg
        # Vertical and horizontal angles to where the beam is pointing.
        beam_theta, beam_phi = self.beam_angles[beam_id]
        if type(phis) == list:
            aw = []
            for phi in phis:
                # TODO: Vertical alignment. 
                aw.append(self.calc_array_factor(theta=np.deg2rad(beam_theta), phi=np.deg2rad(phi-beam_phi)))
        else:
            aw = self.calc_array_factor(theta=np.deg2rad(beam_theta), phi=np.deg2rad(beam_phi))
        return aw, self.beam_angles[beam_id]
        
        # aw = self.calc_array_factor(np.deg2rad(theta), np.deg2rad(phi))
        # # # J.2: Uncomment to activate Jacek's implementation. {{{
        # # n = self.nx * self.ny * self.nz
        # # a = np.ones(n)/np.sqrt(n) #normalize the output
        # # aw = a*self.codebook[beam_id, :]
        # # # }}}
        # return aw, self.beam_angles[beam_id]