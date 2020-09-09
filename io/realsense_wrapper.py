import numpy as np

import pyrealsense2 as rs


class RealSenseWrapper:

    def _convert_depth_frame_to_pointcloud(self, depth_image):
        xyz = self.r * depth_image.reshape(depth_image.shape[0], depth_image.shape[1], 1)
        bad_depth = depth_image == 0
        xyz[bad_depth] = np.nan

        return xyz

    @staticmethod
    def _extinsics2tform(ext):
        tform = np.eye(4)
        tform[:3, :3] = np.asanyarray(ext.rotation).reshape(3, 3)
        tform[:3, -1] = np.asanyarray(ext.translation)
        return tform

    def __del__(self):
        self.pipeline.stop()

    def __init__(self, origin='color'):
        self.pipeline = rs.pipeline()
        cfg = self.pipeline.start()
        profile = {'depth': cfg.get_stream(rs.stream.depth), 'color': cfg.get_stream(rs.stream.color)}

        # self.depth_to_color_extrin = self._extinsics2tform(profile['depth'].get_extrinsics_to(profile['color']))

        if origin == 'color':
            self.aligner = rs.align(rs.stream.color)
            height = profile['color'].as_video_stream_profile().height()
            width = profile['color'].as_video_stream_profile().width()

            self.intrin = profile['color'].as_video_stream_profile().intrinsics

        elif origin == 'depth':
            self.aligner = rs.align(rs.stream.depth)
            height = profile['depth'].as_video_stream_profile().height()
            width = profile['depth'].as_video_stream_profile().width()
            self.intrin = profile['depth'].as_video_stream_profile().intrinsics

        else:
            raise RuntimeError("Unknown origin {}".format(origin))

        self.depth_scale = cfg.get_device().first_depth_sensor().get_depth_scale()

        # data structure for faster pcl creation
        nx = np.linspace(0, width - 1, width)
        ny = np.linspace(0, height - 1, height)
        u, v = np.meshgrid(nx, ny)
        rx = (u - self.intrin.ppx) / self.intrin.fx
        ry = (v - self.intrin.ppy) / self.intrin.fy
        self.r = np.stack((rx, ry, rx * 0 + 1), axis=2)

    def get_xyzirgb(self):
        frameset = self.pipeline.wait_for_frames()
        frameset = self.aligner.process(frameset)
        frame = dict()
        frame['depth'] = frameset.get_depth_frame()
        frame['color'] = frameset.get_color_frame()
        frame['ir'] = frameset.get_infrared_frame()

        if any([not x for x in frame]):
            return None

        depth_image = np.asanyarray(frame['depth'].as_frame().get_data()) * self.depth_scale
        color_image = np.asanyarray(frame['color'].as_frame().get_data())
        ir_image = np.asanyarray(frame['ir'].as_frame().get_data())
        pts = self._convert_depth_frame_to_pointcloud(depth_image)
        xyzrgb = np.concatenate([pts, color_image/255], axis=2)
        return xyzrgb, ir_image

##
