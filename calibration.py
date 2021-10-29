
class StructureCalibration:
    def __init__(self, structure_name, screen_center, delta_gap, structure_center):
        self.structure_name
        self.screen_center = screen_center,
        self.delta_gap = delta_gap
        self.structure_center = structure_center

    def gap_and_beam_position_from_meta(self, meta_data):
        gap = meta_data[self.structure_center+':GAP']*1e-3 + self.delta_gap
        beam_position = -(meta_data[self.structure_center+':CENTER']*1e-3 - self.structure_center)
        return gap, beam_position

