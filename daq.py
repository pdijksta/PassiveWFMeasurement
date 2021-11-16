import itertools
import time
import numpy as np
#import logging
import datetime

import pyscan
from cam_server import PipelineClient, CamClient
from cam_server.utils import get_host_port_from_stream_address
from bsread import source, SUB
from epics import caget, caput

import config

def get_readables(beamline):
    return [
            'bs://image',
            'bs://x_axis',
            'bs://y_axis',
            config.beamline_chargepv[beamline],
            ]

def pyscan_result_to_dict(readables, result, scrap_bs=False):
    """
    Excpects a nested list of order 3.
    Level 1 is the scan index.
    Level 2 is the number of images per scan index (unless this number is 1 in which case this level does not exist).
    Level 3 is the number of readables.

    Returns a shuffled version that takes the form of the dictionary, with the readables as keys.
    """

    output = {}

    for nR, readable in enumerate(readables):
        readable_output1 = []
        for level_scan in result:
            readable_output2 = []
            for level_image in level_scan:
                readable_output2.append(level_image[nR])
            readable_output1.append(readable_output2)

        if scrap_bs and hasattr(readable, 'startswith') and readable.startswith('bs://'):
            readable2 = readable[5:]
        else:
            readable2 = readable

        try:
            output[readable2] = np.array(readable_output1)
        except:
            output[readable2] = readable_output1

    return output

def get_undulator_K(beamline):
    pvs = config.beamline_undulators[beamline]
    return pvs, np.array([caget(pv) for pv in pvs])

def destroy_lasing(beamline, dry_run, max_deltaK=0.2):
    pvs, old_vals = get_undulator_K(beamline)
    randoms = np.random.rand(len(pvs)) - 0.5
    new_vals = old_vals + randoms * max_deltaK
    for pv, new_val in zip(pvs, new_vals):
        if dry_run:
            print('I would caput %s %.6f' % (pv, new_val))
        else:
            caput(pv, new_val)
    return pvs, old_vals, new_vals

def restore_lasing(pvs, vals, dry_run):
    for pv, val in zip(pvs, vals):
        if dry_run:
            print('I would caput %s %.6f' % (pv, val))
        else:
            caput(pv, val)

def get_images(screen, n_images, beamline, dry_run=None):
    if dry_run:
        screen = 'simulation'
    print('Start get_images for screen %s, %i images, beamline %s' % (screen, n_images, beamline))

    def dummy_func(*args):
        pass

    meta_dict_1 = get_meta_data(screen, dry_run, beamline)
    #positioner = pyscan.BsreadPositioner(n_messages=n_images)
    positioner = pyscan.VectorPositioner([0])
    writables = [pyscan.function_value(dummy_func, 'dummy')]
    settings = pyscan.scan_settings(settling_time=0.01, measurement_interval=0.2, n_measurements=n_images)

    pipeline_client = PipelineClient("http://sf-daqsync-01:8889/")
    cam_instance_name = str(screen) + "_sp1"
    stream_address = pipeline_client.get_instance_stream(cam_instance_name)
    stream_host, stream_port = get_host_port_from_stream_address(stream_address)

    # Configure bsread
    pyscan.config.bs_default_host = stream_host
    pyscan.config.bs_default_port = stream_port

    #logging.getLogger("mflow.mflow").setLevel(logging.ERROR)

    readables = get_readables(beamline)

    raw_output = pyscan.scan(positioner=positioner, readables=readables, settings=settings, writables=writables)

    # Bugged?
    # output = [[x] for x in raw_output]
    output = raw_output

    result_dict = pyscan_result_to_dict(readables, output, scrap_bs=True)

    for ax in ['x_axis', 'y_axis']:
        arr = result_dict[ax]*1e-6 # convert to m
        if len(arr.shape) == 3:
            result_dict[ax+'_m'] = arr[0,0,:]
        elif len(arr.shape) == 2:
            result_dict[ax+'_m'] = arr[0,:]
        else:
            raise ValueError('Unexpected', len(arr.shape))

    meta_dict_2 = get_meta_data(screen, dry_run, beamline)
    output_dict = {
            'pyscan_result': result_dict,
            'meta_data_begin': meta_dict_1,
            'meta_data_end': meta_dict_2,
            }

    print('End get_images')
    return output_dict

def data_structure_offset(structure, offset_range, screen, n_images, dry_run, beamline):

    print('Start data_structure_offset for structure %s, screen %s, beamline %s, dry_run %s' % (structure, screen, beamline, dry_run))
    meta_dict_1 = get_meta_data(screen, dry_run, beamline)

    pipeline_client = PipelineClient('http://sf-daqsync-01:8889/')
    offset_pv = structure+':CENTER'

    current_val = caget(offset_pv+'.RBV')

    # Start from closer edge of scan
    if abs(current_val - offset_range[0]) > abs(current_val - offset_range[-1]):
        offset_range = offset_range[::-1]

    positions = offset_range * 1e3 # convert to mm
    positioner = pyscan.VectorPositioner(positions=positions.tolist())

    def dummy_func(*args):
        print('I would set %s to' % offset_pv, args)

    if dry_run:
        screen = 'simulation'
        writables = [pyscan.function_value(dummy_func, 'dummy')]
    else:
        writables = [pyscan.epics_pv(pv_name=offset_pv, readback_pv_name=offset_pv+'.RBV', tolerance=0.005)]

    cam_instance_name = screen + '_sp1'
    stream_address = pipeline_client.get_instance_stream(cam_instance_name)
    stream_host, stream_port = get_host_port_from_stream_address(stream_address)

    # Configure bsread
    pyscan.config.bs_default_host = stream_host
    pyscan.config.bs_default_port = stream_port

    #logging.getLogger('mflow.mflow').setLevel(logging.ERROR)

    settings = pyscan.scan_settings(settling_time=1, n_measurements=n_images, write_timeout=60)

    readables = get_readables(beamline)

    raw_output = pyscan.scan(positioner=positioner, readables=readables, settings=settings, writables=writables)
    result_dict = pyscan_result_to_dict(readables, raw_output, scrap_bs=True)
    #import pdb; pdb.set_trace()
    for ax in ['x_axis', 'y_axis']:
        arr = result_dict[ax]*1e-6
        if len(arr.shape) == 3:
            result_dict[ax+'_m'] = arr[0][0]
        elif len(arr.shape) == 2:
            result_dict[ax+'_m'] = arr[0]
        else:
            raise ValueError('Unexpected', len(arr.shape))

    meta_dict_2 = get_meta_data(screen, dry_run, beamline)

    output = {
            'pyscan_result': result_dict,
            'streaker_offsets': offset_range,
            'screen': screen,
            'n_images': n_images,
            'dry_run': dry_run,
            'structure': structure,
            'meta_data_begin': meta_dict_1,
            'meta_data_end': meta_dict_2,
            }
    print('End data_structure_offset')
    return output

def move_pv(pv, value, timeout, tolerance):
    caput(pv, value)
    step_seconds = 0.1
    max_step = timeout // step_seconds
    for step in range(max_step):
        current_value = caget(pv+'.RBV')
        if abs(current_value - value) < tolerance:
            break
        else:
            time.sleep(step_seconds)
        if step % 10 == 0:
            caput(pv, value)
    else:
        raise ValueError('Pv %s should be %e, is: %e after %f seconds!' % (pv, value, current_value, timeout))

def bpm_data_streaker_offset(streaker, offset_range, screen, n_images, dry_run, beamline):
    print('Start bpm_data_streaker_offset for streaker %s, screen %s, beamline %s, dry_run %s' % (streaker, screen, beamline, dry_run))
    meta_dict_1 = get_meta_data(screen, dry_run, beamline)

    x_axis, y_axis = get_axis(screen)

    result_dict = {'x_axis_m': x_axis, 'y_axis_m': y_axis}

    offset_pv = streaker+':CENTER'

    # Start from closer edge of scan
    current_val = caget(offset_pv+'.RBV')
    if abs(current_val - offset_range[0]) > abs(current_val - offset_range[-1]):
        offset_range = offset_range[::-1]

    offset_range_mm = offset_range * 1e3

    result_dict['image'] = np.zeros([len(offset_range), n_images, len(y_axis), len(x_axis)], dtype=np.uint16)

    bpm_channels = config.beamline_bpm_pvs[beamline]
    charge_channels = config.beamline_charge_pvs_bsread[beamline]

    for bpm_channel in bpm_channels+charge_channels:
        result_dict[bpm_channel] = np.zeros([len(offset_range), n_images])

    for n_offset, offset_mm in enumerate(offset_range_mm):
        if dry_run:
            print('I would move %s to %f' % (offset_pv, offset_mm))
            time.sleep(1)
        else:
            move_pv(offset_pv, offset_mm, 60, 1e-3)
        image_dict = get_images_and_bpm(screen, n_images, beamline, False, False, False, x_axis, y_axis, dry_run)['pyscan_result']
        for key in image_dict.keys():
            if key in result_dict:
                result_dict[key][n_offset] = image_dict[key]

    meta_dict_2 = get_meta_data(screen, dry_run, beamline)
    output = {
            'pyscan_result': result_dict,
            'streaker_offsets': offset_range,
            'screen': screen,
            'n_images': n_images,
            'dry_run': dry_run,
            'streaker': streaker,
            'meta_data_begin': meta_dict_1,
            'meta_data_end': meta_dict_2,
            }
    print('End bpm_data_streaker_offset')
    return output

def get_axis(screen):
    camera_client = CamClient()
    camera_stream_adress = camera_client.get_instance_stream(screen)
    host, port = get_host_port_from_stream_address(camera_stream_adress)

    max_tries = 3
    for _try in range(max_tries):
        try:
            with source(host=host, port=port, mode=SUB) as stream:
                data = stream.receive()
                print('Received image')

                x_axis = np.array(data.data.data['x_axis'].value)*1e-6
                y_axis = np.array(data.data.data['y_axis'].value)*1e-6
        except Exception as e:
            print('Try %i' % _try, e)
            pass
        else:
            break
    else:
        raise

    return x_axis, y_axis

def get_images_and_bpm(screen, n_images, beamline, axis=True, print_=True, include_meta_data=True, x_axis=None, y_axis=None, dry_run=False):

    if print_:
        print('Start get_images_and_bpm for screen %s, %i images, beamline %s' % (screen, n_images, beamline))

    if include_meta_data:
        meta_dict_1 = get_meta_data(screen, dry_run, beamline)
    else:
        meta_dict_1 = None

    result_dict = {}

    if axis:
        x_axis, y_axis = get_axis(screen)
        result_dict['x_axis_m'] = x_axis
        result_dict['y_axis_m'] = y_axis

    bpm_channels = config.beamline_bpm_pvs[beamline]
    charge_channels = config.beamline_charge_pvs_bsread[beamline]
    image_pv = screen+':FPICTURE'
    channels = bpm_channels + [image_pv] + charge_channels

    images = np.zeros([n_images, len(y_axis), len(x_axis)], dtype=np.uint16)
    bpm_values = np.zeros([len(bpm_channels+charge_channels), n_images])
    pulse_ids = np.zeros(n_images)

    max_tries = 3
    for _try in range(max_tries):
        try:
            with source(channels=channels) as stream:
                for n_image in range(n_images):
                    msg = stream.receive()
                    pulse_ids[n_image] = msg.data.pulse_id
                    for n_bpm, bpm_channel in enumerate(bpm_channels+charge_channels):
                        bpm_values[n_bpm, n_image] = msg.data.data[bpm_channel].value
                    if not dry_run:
                        images[n_image] = msg.data.data[image_pv].value

            result_dict['image'] = images
            result_dict['pulse_id'] = pulse_ids
            for n_bpm, bpm_channel in enumerate(bpm_channels+charge_channels):
                result_dict[bpm_channel] = bpm_values[n_bpm]
        except Exception as e:
            print('Try %i' % _try, e)
            raise
        else:
            break
    else:
        raise

    if include_meta_data:
        meta_dict_2 = get_meta_data(screen, dry_run, beamline)
    else:
        meta_dict_2 = None

    output_dict = {
            'pyscan_result': result_dict,
            'meta_data_begin': meta_dict_1,
            'meta_data_end': meta_dict_2,
            }

    if print_:
        print('End get_images')

    return output_dict

def get_quad_strengths(beamline):
    quads = config.beamline_quads[beamline]

    k1l_dict = {}
    for quad in quads:
        k1l_dict[quad] = caget(quad.replace('.', '-')+':K1L-SET')
    energy_pv = config.beamline_energypv[beamline]
    k1l_dict[energy_pv] = caget(energy_pv)
    return k1l_dict

def get_meta_data(screen, dry_run, beamline):
    all_structures = config.all_structures
    meta_dict = {}
    for streaker, suffix1, suffix2 in itertools.product(all_structures, [':GAP', ':CENTER'], ['', '.RBV']):
        pv = streaker+suffix1+suffix2
        meta_dict[pv] = caget(pv)
    meta_dict.update({x: caget(x) for x in config.beamline_chargepv.values()})

    energy_pv = screen+':ENERGY-OP'
    if dry_run:
        meta_dict[energy_pv] = 6000
    else:
        meta_dict[energy_pv] = caget(energy_pv)

    k1l_dict = get_quad_strengths(beamline)
    meta_dict.update(k1l_dict)
    meta_dict['time'] = str(datetime.datetime.now())
    for gas_monitor_energy_pv in config.gas_monitor_pvs.values():
        meta_dict[gas_monitor_energy_pv] = caget(gas_monitor_energy_pv)
    return meta_dict

