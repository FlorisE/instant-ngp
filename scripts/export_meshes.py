import argparse
import json
import os

from common import *

import pyngp as ngp # noqa

from tqdm import trange

def parse_args():
    parser = argparse.ArgumentParser(description='Export meshes for an entire folder')
    parser.add_argument('folder')
    parser.add_argument('--iterations', type=int, default=50000)
    parser.add_argument('--save_iterations', type=int, default=10000)
    parser.add_argument('--optimize_extrinsics', type=bool, default=True)
    parser.add_argument('--save_optimized_extrinsics', type=bool, default=True)
    parser.add_argument('--gui', action='store_true')
    return parser.parse_args()

def process(folder, iterations, save_iterations, optimize_extrinsics, save_optimized_extrinsics, gui):
    for item in os.listdir(folder):
        if os.path.isdir(folder + os.sep + item):
            testbed = ngp.Testbed()
            if gui:
                testbed.init_window(1920, 1280)
            testbed.shall_train = True
            print(f'Optimize extrinsics: {optimize_extrinsics}')
            nerf_folder = folder + os.sep + item
            testbed.load_file(nerf_folder)
            target_json = None
            for nerf_folder_file in os.listdir(nerf_folder):
                if nerf_folder_file.endswith('.json'):
                    target_json = nerf_folder_file
                    break
            if target_json is None:
                print(f'No JSON file found in {nerf_folder}')
                continue

            testbed.nerf.training.optimize_extrinsics = optimize_extrinsics
            for i in trange(iterations):
                testbed.frame()
                if (i + 1) % save_iterations == 0:
                    res = 1024
                    thresh = 2.5
                    if iterations == save_iterations:
                        target = nerf_folder + os.sep + item + '.obj'
                    else:
                        target = nerf_folder + os.sep + item + str(i) + '.obj'
                    print(f"Writing mesh to {target}")
                    if os.path.exists(target):
                        os.remove(target)
                    mesh_it = testbed.compute_and_save_marching_cubes_mesh
                    mesh_it(target,
                            [res, res, res],
                            thresh=thresh,
                            generate_uvs_for_obj_file=True)
            if save_optimized_extrinsics:
                optimized_extrinsics = []
                for i in range(testbed.nerf.training.dataset.n_images):
                    extrinsics = testbed.nerf.training.get_camera_extrinsics(i)
                    optimized_extrinsics.append({'transform_matrix': extrinsics.tolist()})
                with open(nerf_folder + os.sep + target_json, 'r+') as f:
                    input_transforms_json = json.load(f)
                input_transforms_json['frames'] = sorted(input_transforms_json['frames'], key=lambda frame: frame['file_path'])
                for frame, extrinsics in zip(input_transforms_json['frames'], optimized_extrinsics):
                    frame['transform_matrix'] = extrinsics['transform_matrix']
                    frame['transform_matrix'].append([0, 0, 0, 1])
                with open(nerf_folder + os.sep + target_json, 'w') as f:
                    json.dump(input_transforms_json, f, indent=2)

if __name__ == '__main__':
    args = parse_args()

    if args.save_iterations > args.iterations:
        args.save_iterations = args.iterations
    process(args.folder, args.iterations, args.save_iterations, args.optimize_extrinsics, args.save_optimized_extrinsics, args.gui)
