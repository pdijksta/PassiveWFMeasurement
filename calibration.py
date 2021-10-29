
class StructureCalibration:
    def __init__(self, structure_name, screen_center, delta_gap, structure_center):
        self.structure_name = structure_name
        self.screen_center = screen_center,
        self.delta_gap = delta_gap
        self.structure_center = structure_center

    def gap_and_beam_position_from_meta(self, meta_data):
        gap0 = meta_data[self.structure_name+':GAP']*1e-3
        gap = gap0 + self.delta_gap
        structure_center = meta_data[self.structure_name+':CENTER']*1e-3
        beam_position = -(structure_center - self.structure_center)
        distance = gap/2. - abs(beam_position)
        if distance < 0:
            raise ValueError('Distance between beam and gap is negative')

        return {
                'gap0': gap0,
                'gap': gap,
                'structure_center': structure_center,
                'beam_position': beam_position,
                'distance': distance,
                }

