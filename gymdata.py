from torchmultimodal.transforms.video_transform import VideoTransform
import os
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset
import torchvision.transforms as T


class GymDataset(Dataset):
    def __init__(self, path_list, target_frames=16, image_size=180, augment=False):
        self.video_paths = path_list
        self.augment = augment
        self.vt = VideoTransform()
        self.vt.time_samples = target_frames 
        self.vt.resize_shape = (image_size,image_size)
        
        self.CLASS_LABELS = {
            "barbell biceps curl": 0,
            "bench press": 1,
            "chest fly machine": 2,
            "deadlift": 3,
            "decline bench press": 4,
            "hammer curl": 5,
            "hip thrust": 6,
            "incline bench press": 7,
            "lat pulldown": 8,
            "lateral raise": 9,
            "leg extension": 10,
            "leg raises": 11,
            "plank": 12,
            "pull Up": 13,
            "push-up": 14,
            "romanian deadlift": 15,
            "russian twist": 16,
            "shoulder press": 17,
            "squat": 18,
            "t bar row": 19,
            "tricep dips": 20,
            "tricep Pushdown": 21
        }
        
        if self.augment:
            self.transform = T.Compose([
                T.ToPILImage(),
                T.RandomApply([
                    T.RandomRotation(degrees=(-30,30))
                ], p=0.80),
                T.RandomHorizontalFlip(p=0.4),
                T.RandomVerticalFlip(p=0.4),
                T.ToTensor()
            ])
        else:
            self.transform = T.Compose([
                T.ToPILImage(),
                T.ToTensor()

            ])
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.CLASS_LABELS[os.path.basename(os.path.dirname(video_path))]
        
        video, _, _ = torchvision.io.read_video(video_path,pts_unit='sec')
        video = self.vt(video.unsqueeze(0))
        video = video.squeeze(0)
        video = video.permute(1, 0, 2, 3)  # T, C, H, W
        
        return video.permute(1,0,2,3), label 