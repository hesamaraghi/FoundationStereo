import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.cm as cm
import torch 
import torch.nn as nn 
from glob import glob
from matplotlib import pyplot as plt


class Pixel_Projector():
    """
    Match pixels between event cameras and the dpeth caemra. 
    """
    def __init__(self, calibration_path, data_path):
        """
        Read intrinsic matrix
        """
        #### Read DRGB intrinsic matrix
        self.DRGB_intrinsic_matrix, self.DRGB_distortion_matrix = self.read_intrinsic_calibration(calibration_path, 'DRGB')

        #### Read EVENT0 intrinsic matrix
        self.EVENT0_intrinsic_matrix, self.EVENT0_distortion_matrix = self.read_intrinsic_calibration(calibration_path, 'EVENT0')

        #### Read EVENT1 intrinsic matrix
        self.EVENT1_intrinsic_matrix, self.EVENT1_distortion_matrix = self.read_intrinsic_calibration(calibration_path, 'EVENT1')

        """
        Read extrinsic matrix
        """
        #### Read EVENT0 -- DRGB extrinsic matrix
        self.rotation_matrix_EVENT0_DRGB, self.translation_matrix_EVENT0_DRGB, _, _, self.stereoMapDRGB_X, self.stereoMapDRGB_Y = self.read_extrinsic_calibration(calibration_path, camA='EVENT0', camB='DRGB')

        #### Read EVENT0 -- EVENT1 extrinsic matrix
        self.rotation_matrix_EVENT0_EVENT1, self.translation_matrix_EVENT0_EVENT1, self.stereoMapEVENT0_X, self.stereoMapEVENT0_Y, self.stereoMapEVENT1_X, self.stereoMapEVENT1_Y = self.read_extrinsic_calibration(calibration_path, camA='EVENT0', camB='EVENT1')

        #### Read DRGB -- EVENT0 extrinsic matrix
        self.rotation_matrix_DRGB_EVENT0, self.translation_matrix_DRGB_EVENT0, _, _, _, _ = self.read_extrinsic_calibration(calibration_path, camA='DRGB', camB='EVENT0')


        """
        Read data
        *** Please change the data path to your own ***
        """
        #### DEPTH data path
        # self.DEPTH_data_path = os.path.join(data_path, 'extractIMAGE', 'DEPTH')
        # DEPTH_ts = []
        # for p in glob(os.path.join(self.DEPTH_data_path, '*.tif')):
        #     DEPTH_ts.append(float(os.path.basename(p)[:-4]))
        # self.DEPTH_ts = np.array(DEPTH_ts, dtype=np.float64)

        #### EVENT0 data path
        self.EVENT0_data_path = os.path.join(data_path, 'EVENT0', 'e2calib')
        EVENT0_ts = []
        self.EVENT0_data_path_list = sorted(glob(os.path.join(self.EVENT0_data_path, '*.png')))
        for p in self.EVENT0_data_path_list:
            base_name = os.path.basename(p)[:-4]
            ts_str = base_name[:-9] + '.' + base_name[-9:]
            EVENT0_ts.append(float(ts_str))
        self.EVENT0_ts = np.array(EVENT0_ts, dtype=np.float64)

        #### EVENT1 data path
        self.EVENT1_data_path = os.path.join(data_path, 'EVENT1', 'e2calib')
        EVENT1_ts = []
        self.EVENT1_data_path_list = sorted(glob(os.path.join(self.EVENT1_data_path, '*.png')))
        for p in self.EVENT1_data_path_list:
            base_name = os.path.basename(p)[:-4]
            ts_str = base_name[:-9] + '.' + base_name[-9:]
            EVENT1_ts.append(float(ts_str))
        self.EVENT1_ts = np.array(EVENT1_ts, dtype=np.float64)


    def read_intrinsic_calibration(self, calibration_path, cam):
        intrinsic_path = os.path.join(calibration_path, 'intrinsic_calibration_results')
        _intrinsic_file = os.path.join(intrinsic_path, f'{cam}_intrinsic.xml')
        assert os.path.isfile(_intrinsic_file), f'{_intrinsic_file} does not exist!'
        _intrinsic_loader = cv2.FileStorage(_intrinsic_file, cv2.FileStorage_READ)
        _intrinsic_matrix = _intrinsic_loader.getNode('Intrinsic_Matrix').mat()
        _distortion_matrix = _intrinsic_loader.getNode('Distortion_Matrix').mat()

        return _intrinsic_matrix, _distortion_matrix
    

    def read_extrinsic_calibration(self, calibration_path, camA, camB):
        extrinsic_path = os.path.join(calibration_path, 'stereo_calibration_results')
        _extrinsic_file = os.path.join(extrinsic_path, f'{camA}_{camB}_stereo_calibration.xml')
        assert os.path.isfile(_extrinsic_file), f'{_extrinsic_file} does not exist'
        _extrinsic_loader = cv2.FileStorage(_extrinsic_file, cv2.FileStorage_READ)
        _rotation_matrix = _extrinsic_loader.getNode('Rotation_Matrix').mat()
        _translation_matrix = _extrinsic_loader.getNode('Translation_Matrix').mat()
        _stereoMapA_X = _extrinsic_loader.getNode('stereoMapA_X').mat()
        _stereoMapA_Y = _extrinsic_loader.getNode('stereoMapA_Y').mat()
        _stereoMapB_X = _extrinsic_loader.getNode('stereoMapB_X').mat()
        _stereoMapB_Y = _extrinsic_loader.getNode('stereoMapB_Y').mat()

        return _rotation_matrix, _translation_matrix, _stereoMapA_X, _stereoMapA_Y, _stereoMapB_X, _stereoMapB_Y    


    def fast_reproject(self, depth_map, cameraB_instrinsic_matrix, translation_matrix, rotation_matrix):
        # create a meshgrid of pixel coordinates in DRGB (cam A)
        x_A, y_A = np.meshgrid(np.arange(1280), np.arange(720))

        # convert 2d coordiantes into normalized 3d
        Z = depth_map
        X = (x_A - self.DRGB_intrinsic_matrix[0, 2]) * Z / self.DRGB_intrinsic_matrix[0, 0]
        Y = (y_A - self.DRGB_intrinsic_matrix[1, 2]) * Z / self.DRGB_intrinsic_matrix[1, 1]

        # Stack X, Y, and Z to form 3D points in Camera A's coordinate frame
        points_3D_A = np.vstack((X.flatten(), Y.flatten(), Z.flatten())).T

        # Apply rotation and translation to get points in Camera B's coordinate system
        points_3D_B = np.dot(rotation_matrix, points_3D_A.T) + translation_matrix.reshape(3, 1)
        points_3D_B = points_3D_B.T

        # Project 3D points in Camera B's coordinates into Camera B's image plane
        points_2D_B = np.dot(cameraB_instrinsic_matrix, points_3D_B.T)

        # Normalize to get pixel coordinates (divide by the Z coordinate)
        points_2D_B[0, :] /= points_2D_B[2, :]
        points_2D_B[1, :] /= points_2D_B[2, :]

        # Extract pixel coordinates in Camera B's image
        x_B = points_2D_B[0, :].reshape(720, 1280).astype(np.float32)
        y_B = points_2D_B[1, :].reshape(720, 1280).astype(np.float32)

        # Create a mask to identify valid pixels
        mask = (x_B >= 0) & (x_B < 1280) & (y_B >= 0) & (y_B < 720)
        
        # Create maps for remapping Camera A's frame to Camera B's image coordinates
        map_x_B = np.where(mask, x_B, 0)  # Only valid x_B coordinates
        map_y_B = np.where(mask, y_B, 0)  # Only valid y_B coordinates

        # Optionally, you can clip the maps to the valid range of Camera B's image size
        map_x_B = np.clip(map_x_B, 0, 1280 - 1)
        map_y_B = np.clip(map_y_B, 0, 720 - 1)

        # Warp Camera A's image to Camera B's perspective
        # Create an empty image for the projected output
        new_depth = np.zeros_like(depth_map)

        # Only warp the valid pixels from frame_A_undistorted to frame_A_warped_to_B
        new_depth[y_B[mask].astype(int), x_B[mask].astype(int)] = depth_map[y_A[mask].astype(int), x_A[mask].astype(int)]

        # post processing
        new_depth = self.interpolation_after_projection(new_depth)

        return new_depth


    def interpolation_after_projection(self, image):
        # convert to torch tensor
        image = torch.from_numpy(image).reshape(1, 1, 720, 1280).type(torch.float32)
        # maxpooling to vanish the 0 value in the projected image
        pool = nn.MaxPool2d(kernel_size=4, stride=1)
        image = pool(image)
        image = image.reshape(image.shape[2], image.shape[3])
        # to array
        image = np.array(image)
        # resize back to (1280, 720)
        projected_frame = cv2.resize(image, (1280, 720), interpolation=cv2.INTER_LINEAR)

        return projected_frame


    def undistort_image(self, img, intinsic_matrix, distortion_matrix):
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(intinsic_matrix,
                                                          distortion_matrix,
                                                          (1280,720),
                                                          1,
                                                          (1280,720))
        dst = cv2.undistort(img, intinsic_matrix, distortion_matrix, None, newcameramtx)

        return dst, newcameramtx


    def rectify(self, ts):
        """
        read images
        """ 
        # synced(nearst) event0
        idx = np.abs(self.EVENT0_ts-ts).argmin()
        event0_ts = self.EVENT0_ts[idx]
        event0_path = self.EVENT0_data_path_list[idx]
        event0_image = cv2.imread(event0_path)
        
        # synced(nearst) event1
        idx = np.abs(self.EVENT1_ts-ts).argmin()
        event1_ts = self.EVENT1_ts[idx]
        event1_path = self.EVENT1_data_path_list[idx]
        event1_image = cv2.imread(event1_path)
  
        """
        project depth to the FOV of EVENT0
        """
        # undistort image
        event0_image, self.EVENT0_intrinsic_matrix = self.undistort_image(event0_image,
                                                                          self.EVENT0_intrinsic_matrix,
                                                                          self.EVENT0_distortion_matrix)
        event1_image, self.EVENT1_intrinsic_matrix = self.undistort_image(event1_image,
                                                                          self.EVENT1_intrinsic_matrix,
                                                                          self.EVENT1_distortion_matrix)

        """
        Rectify EVENT0, EVENT1, and projected depth
        """
        # rectify images
        rectified_event0_image = cv2.remap(event0_image,
                                 self.stereoMapEVENT0_X,
                                 self.stereoMapEVENT0_Y,
                                 cv2.INTER_LINEAR)
        rectified_event1_image = cv2.remap(event1_image,
                                 self.stereoMapEVENT1_X,
                                 self.stereoMapEVENT1_Y,
                                 cv2.INTER_LINEAR)

        return rectified_event0_image, rectified_event1_image

if __name__ == '__main__':
    
    disk = 'NSEK'
    if pc == 'mac':
        disk = os.path.join('/Volumes/', disk)
    elif pc == 'dell':
        disk = os.path.join('/media/chengming/', disk)
    kitchen = 'K_FCM'
    activity = 'cereal_bowl'

    projector = Pixel_Projector(calibration_path=os.path.join(disk, kitchen, 'calibration'),
                                data_path=os.path.join(disk, kitchen, activity))
    
    rectified_event0_image, rectified_event1_image = projector.rectify(1704555846.9445188)



