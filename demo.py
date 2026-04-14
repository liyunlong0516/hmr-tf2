"""
Demo of HMR.

Note that HMR requires the bounding box of the person in the image. The best performance is obtained when max length of the person in the image is roughly 150px. 

When only the image path is supplied, it assumes that the image is centered on a person whose length is roughly 150px.
Alternatively, you can supply output of the openpose to figure out the bbox and the right scale factor.

Sample usage:

# On images on a tightly cropped image around the person
python -m demo --img_path data/im1963.jpg
python -m demo --img_path data/coco1.png

# On images, with openpose output
python -m demo --img_path data/random.jpg --json_path data/random_keypoints.json
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
from absl import flags
import numpy as np

# Use non-interactive backend for matplotlib
import matplotlib
matplotlib.use('Agg')

import skimage.io as io
import tensorflow as tf

from src.util import renderer as vis_util
from src.util import image as img_util
from src.util import openpose as op_util
import src.config
from src.RunModel import RunModel

flags.DEFINE_string('img_path', 'data/im1963.jpg', 'Image to run')
flags.DEFINE_string(
    'json_path', None,
    'If specified, uses the openpose output to crop the image.')


def save_mesh_to_obj(verts, faces, output_path='output_mesh.obj', apply_transform=True):
    """
    Save mesh vertices and faces to OBJ file format.
    
    Args:
        verts: N x 3 array of vertex positions
        faces: M x 3 array of face indices
        output_path: Path to save the OBJ file
        apply_transform: If True, apply coordinate transformation to make model upright and facing forward
    """
    # Apply coordinate transformation if requested
    if apply_transform:
        # SMPL uses: Y-up, Z-forward (camera coordinate)
        # Standard 3D software uses: Y-up, -Z-forward or Z-up, -Y-forward
        # Transform: rotate 180 degrees around X axis to flip Y and Z
        # This makes the model upright and facing forward
        transform_matrix = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ])
        verts_transformed = np.dot(verts, transform_matrix.T)
    else:
        verts_transformed = verts
    
    with open(output_path, 'w') as f:
        # Write vertices
        for v in verts_transformed:
            f.write('v %.6f %.6f %.6f\n' % (v[0], v[1], v[2]))
        
        # Write faces (OBJ uses 1-based indexing)
        for face in faces:
            f.write('f %d %d %d\n' % (face[0] + 1, face[1] + 1, face[2] + 1))
    
    print(f'Mesh saved to {output_path}')
    print(f'  Vertices: {len(verts_transformed)}')
    print(f'  Faces: {len(faces)}')
    if apply_transform:
        print('  Applied coordinate transformation: Y-up, -Z-forward (standard 3D)')


def visualize(img, proc_param, joints, verts, cam, save_mesh=False):
    """
    Renders the result in original image coordinate frame.
    """
    cam_for_render, vert_shifted, joints_orig = vis_util.get_original(
        proc_param, verts, cam, joints, img_size=img.shape[:2])

    # Render results
    skel_img = vis_util.draw_skeleton(img, joints_orig)
    rend_img_overlay = renderer(
        vert_shifted, cam=cam_for_render, img=img, do_alpha=True)
    rend_img = renderer(
        vert_shifted, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp1 = renderer.rotated(
        vert_shifted, 60, cam=cam_for_render, img_size=img.shape[:2])
    rend_img_vp2 = renderer.rotated(
        vert_shifted, -60, cam=cam_for_render, img_size=img.shape[:2])

    import matplotlib.pyplot as plt
    # plt.ion()
    plt.figure(1)
    plt.clf()
    plt.subplot(231)
    plt.imshow(img)
    plt.title('input')
    plt.axis('off')
    plt.subplot(232)
    plt.imshow(skel_img)
    plt.title('joint projection')
    plt.axis('off')
    plt.subplot(233)
    plt.imshow(rend_img_overlay)
    plt.title('3D Mesh overlay')
    plt.axis('off')
    plt.subplot(234)
    plt.imshow(rend_img)
    plt.title('3D mesh')
    plt.axis('off')
    plt.subplot(235)
    plt.imshow(rend_img_vp1)
    plt.title('diff vp')
    plt.axis('off')
    plt.subplot(236)
    plt.imshow(rend_img_vp2)
    plt.title('diff vp')
    plt.axis('off')
    plt.draw()
    # Save figure instead of showing
    output_path = 'output_result.png'
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f'Result saved to {output_path}')
    plt.close()
    
    # Save mesh to OBJ file if requested
    if save_mesh:
        # Load SMPL faces
        import os.path as osp
        curr_path = osp.dirname(osp.abspath(__file__))
        face_path = osp.join(curr_path, 'src', 'tf_smpl', 'smpl_faces.npy')
        faces = np.load(face_path)
        save_mesh_to_obj(vert_shifted, faces, 'output_mesh.obj')


def preprocess_image(img_path, json_path=None):
    img = io.imread(img_path)
    if img.shape[2] == 4:
        img = img[:, :, :3]

    if json_path is None:
        if np.max(img.shape[:2]) != config.img_size:
            print('Resizing so the max image size is %d..' % config.img_size)
            scale = (float(config.img_size) / np.max(img.shape[:2]))
        else:
            scale = 1.
        center = np.round(np.array(img.shape[:2]) / 2).astype(int)
        # image center in (x,y)
        center = center[::-1]
    else:
        scale, center = op_util.get_bbox(json_path)

    crop, proc_param = img_util.scale_and_crop(img, scale, center,
                                               config.img_size)

    # Normalize image to [-1, 1]
    crop = 2 * ((crop / 255.) - 0.5)

    return crop, proc_param, img


def main(img_path, json_path=None):
    sess = tf.compat.v1.Session()
    model = RunModel(config, sess=sess)

    input_img, proc_param, img = preprocess_image(img_path, json_path)
    # Add batch dimension: 1 x D x D x 3
    input_img = np.expand_dims(input_img, 0)

    # Theta is the 85D vector holding [camera, pose, shape]
    # where camera is 3D [s, tx, ty]
    # pose is 72D vector holding the rotation of 24 joints of SMPL in axis angle format
    # shape is 10D shape coefficients of SMPL
    joints, verts, cams, joints3d, theta = model.predict(
        input_img, get_theta=True)

    # Save mesh by default
    visualize(img, proc_param, joints[0], verts[0], cams[0], save_mesh=True)


if __name__ == '__main__':
    config = flags.FLAGS
    config(sys.argv)
    # Using pre-trained model, change this to use your own.
    config.load_path = src.config.PRETRAINED_MODEL

    config.batch_size = 1

    renderer = vis_util.SMPLRenderer(face_path=config.smpl_face_path)

    main(config.img_path, config.json_path)
