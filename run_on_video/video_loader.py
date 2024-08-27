import torch as th
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
# import ffmpeg
import math
import cv2


def convert_to_float(frac_str):
    try:
        return float(frac_str)
    except ValueError:
        try:
            num, denom = frac_str.split('/')
        except ValueError:
            return None
        try:
            leading, num = num.split(' ')
        except ValueError:
            return float(num) / float(denom)
        if float(leading) < 0:
            sign_mult = -1
        else:
            sign_mult = 1
        return float(leading) + sign_mult * (float(num) / float(denom))


class VideoLoader(Dataset):
    """Pytorch video loader."""

    def __init__(
            self,
            vid_path,
            framerate=1,
            size=112,
            centercrop=False,
            overwrite=False,
            model_version="ViT-B/32",
    ):
        """
        Args:
        """
        self.vid_path = vid_path

        self.centercrop = centercrop
        self.size = size
        self.framerate = framerate
        self.overwrite = overwrite
        self.model_version = model_version


    def __len__(self):
        return 1


    def _get_video_info(self, video_path):

        # probe = ffmpeg.probe(video_path)
        # video_stream = next((stream for stream in probe['streams']
        #                      if stream['codec_type'] == 'video'), None)
        # width = int(video_stream['width'])
        # height = int(video_stream['height'])
        # fps = math.floor(convert_to_float(video_stream['avg_frame_rate']))
        # try:
        #     frames_length = int(video_stream['nb_frames'])
        #     duration = float(video_stream['duration'])
        # except Exception:
        #     frames_length, duration = -1, -1


        info = {
            "duration": duration, 
            "frames_length": frames_length, 
            "fps": fps, 
            "height": height, 
            "width": width, 
            }

        return info


    def _get_output_dim(self, h, w):
        if isinstance(self.size, tuple) and len(self.size) == 2:
            return self.size
        elif h >= w:
            return int(h * self.size / w), self.size
        else:
            return self.size, int(w * self.size / h)


    def __getitem__(self, id):

        video_path = self.vid_path

        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError("Could not open the video file.")
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Total frame count might not be accurate for some codecs
        frames_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate duration
        if original_fps > 0:
            duration = frames_length / original_fps
        else:
            duration = -1
            
        import ipdb; ipdb.set_trace()

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        size = self.size


        try:
            target_fps = self.framerate
            if duration > 0 and duration < 1/target_fps+0.1:
                target_fps = 2/max(int(duration), 1)
                print(duration, target_fps)
        except Exception:
            target_fps = self.framerate


        frame_interval = int(original_fps / target_fps)
        if frame_interval <= 0:
            frame_interval = 1

        frames = []
        frame_index = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if self.centercrop:
                x = int((width - size) / 2.0)
                y = int((height - size) / 2.0)
                frame = frame[y:y+size, x:x+size]
            else:
                frame = cv2.resize(frame, (size, size))

            if frame_index % frame_interval == 0:
                frames.append(frame)

            frame_index += 1
        
        cap.release()
                
        video = np.array(frames, dtype=np.uint8)
        video = th.from_numpy(video.astype('float32'))
        video = video.permute(0, 3, 1, 2)

        # load_flag = os.path.isfile(video_path)
        # if load_flag:
        #     try:
        #         # import pdb; pdb.set_trace()
        #         info = self._get_video_info(video_path)
        #         h, w = info["height"], info["width"]
        #     except Exception:
        #         print('ffprobe failed at: {}'.format(video_path))
        #         return {'video': th.zeros(1), 'input': video_path,'info': {}}
        #     try:
        #         height, width = self._get_output_dim(h, w)
        #         try:
        #             duration = info["duration"]
        #             fps = self.framerate
        #             if duration > 0 and duration < 1/fps+0.1:
        #                 fps = 2/max(int(duration), 1)
        #                 print(duration, fps)
        #         except Exception:
        #             fps = self.framerate

        #         cmd = (
        #             ffmpeg
        #             .input(video_path)
        #             .filter('fps', fps=fps)
        #             .filter('scale', width, height)
        #             # .filter('scale', self.size, self.size)
        #         )
        #         if self.centercrop:
        #             x = int((width - self.size) / 2.0)
        #             y = int((height - self.size) / 2.0)
        #             cmd = cmd.crop(x, y, self.size, self.size)
        #         out, _ = (
        #             cmd.output('pipe:', format='rawvideo', pix_fmt='rgb24')
        #             .run(capture_stdout=True, quiet=True)
        #         )
        #         if self.centercrop and isinstance(self.size, int):
        #             height, width = self.size, self.size
        #         video = np.frombuffer(out, np.uint8).reshape(
        #             [-1, height, width, 3])
        #         video = th.from_numpy(video.astype('float32'))
        #         video = video.permute(0, 3, 1, 2)
        #     except:
        #         return {'video': th.zeros(1), 'input': video_path,'info': {}}
        # else:
        #     video = th.zeros(1)

        return {'video': video, 'input': video_path}
