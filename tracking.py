from . import beam
from . import lattice
from . import wf_model

class Tracker2:
    """
    Parameters:
        beamline - Can be "Aramis" or "Athos".
        screen_name - For example "SARBD02-DSCR050".
        streaker_name - For example "SARUN18-UDCP020".
        meta_data - Look at daq.py for format.
        beam_parameters - Look at config.py for format.
        delta_gap - Calibrated gap.
        structure_center - Structure position for which the beam is centered inside.
        screen_center - Position of the unstreaked beam on the screen.
    """
    def __init__(self, beamline, screen_name, streaker_name, meta_data, beam_parameters, delta_gap, structure_center, screen_center):
        self.meta_data = meta_data
        self.streaker_name = streaker_name
        self.screen_name = screen_name
        self.beamline = beamline
        self.structure_center = structure_center
        self.screen_center = screen_center
        self.lat = lattice.get_beamline_lattice(beamline, meta_data)

        matrix = self.lat.get_matrix(streaker_name, screen_name)
        self.r12 = matrix[0,1]
        self.disp = matrix[2,5]
        self.energy = meta_data[screen_name+':ENERGY-OP']
        self.set_delta_gap(delta_gap, None)
        self.trans_beam = beam.gen_beam4D(**beam_parameters, p_central=self.energy/beam.electron_mass_eV)

    def set_streaker(self, semigap):
        if self.beamline == 'Aramis':
            self.streaker = wf_model.PostAramisStreaker(semigap)
        elif self.beamline == 'Athos':
            self.streaker = wf_model.PostAramisStreaker(semigap)

    def set_delta_gap(self, delta_gap, delta_gap_old):
        if (delta_gap is None) or (delta_gap_old != delta_gap):
            self.delta_gap = delta_gap
            semigap = self.meta_data[self.streaker_name+':GAP']*1e-3/2 + delta_gap
            self.set_streaker(semigap)

    def forward_propagate(self, beam):
        pass


