import numpy as np
#from . import beam
from . import lattice
from . import wf_model

class Tracker:
    """
    Parameters:
        beamline - Can be "Aramis" or "Athos".
        screen_name - For example "SARBD02-DSCR050".
        structure_name - For example "SARUN18-UDCP020".
        meta_data - Look at daq.py for format.
        beam_parameters - Look at config.py for format.
        delta_gap - Calibrated gap.
        structure_center - Structure position for which the beam is centered inside.
        screen_center - Position of the unstreaked beam on the screen.
    """
    def __init__(self, beamline, screen_name, structure_name, meta_data, delta_gap, structure_center, screen_center):
        self.meta_data = meta_data
        self.structure_name = structure_name
        self.screen_name = screen_name
        self.beamline = beamline
        self.structure_center = structure_center
        self.screen_center = screen_center
        self.lat = lattice.get_beamline_lattice(beamline, meta_data)

        self.matrix = self.lat.get_matrix(structure_name.replace('-', '.'), screen_name.replace('-', '.'))
        self.r12 = self.matrix[0,1]
        self.disp = self.matrix[2,5]
        self.energy_eV = meta_data[screen_name+':ENERGY-OP']*1e6
        self.delta_gap = delta_gap
        self.structure = wf_model.get_structure(structure_name)
        self.structure_position = meta_data[structure_name+':CENTER']*1e-3

    def forward_propagate(self, beam, streaking_obj, quad_wake):
        """
        beam: must be beam corresponding to beginning of self.lat
        """
        mat = self.lat.get_matrix(self.lat.element_names[0].replace('-', '.'), self.structure_name.replace('-', '.'))
        beam.linear_propagate(mat)
        wake_time = streaking_obj.time_grid
        energy_eV = beam.energy_eV
        delta_xp_dipole = streaking_obj.convolve(beam.beamProfile, 'Dipole')/energy_eV
        delta_xp_coords_dip = np.interp(beam['t'], wake_time, delta_xp_dipole)
        if quad_wake:
            delta_xp_quadrupole = streaking_obj.convolve(beam.beamProfile, 'Quadrupole')/energy_eV
            delta_xp_coords_quad = np.interp(beam['t'], wake_time, delta_xp_quadrupole)
        else:
            delta_xp_quadrupole = 0.
            delta_xp_coords_quad = 0.

        beam['xp'] += delta_xp_coords_dip + delta_xp_coords_quad

        beam.linear_propagate(self.matrix)
        return beam

    def backward_propagate(self):
        pass


