import torch
from tqdm import tqdm
import os
import numpy as np
import open3d as o3d
import open_clip
from PIL import Image
from torch.nn import functional as F
from pytorch3d.ops import ball_query
from torch_scatter import scatter_mean
import cv2

clip_model, _, clip_preprocess = open_clip.create_model_and_transforms('ViT-L-14-336',pretrained = "openai",device='cuda')

def get_depth_mask(depth):
    depth_tensor = torch.from_numpy(depth).cuda()
    depth_mask = torch.logical_and(depth_tensor > 0, depth_tensor < 20).reshape(-1)
    return depth_mask

def backproject(depth, intrinisc_cam_parameters, extrinsics):
    """
    convert color and depth to view pointcloud
    """
    depth = o3d.geometry.Image(depth)
    pcld = o3d.geometry.PointCloud.create_from_depth_image(depth, intrinisc_cam_parameters, depth_scale=1, depth_trunc=20)
    pcld.transform(extrinsics)
    return pcld

def turn_pixel_to_point(scene_points, frame_id, seq_name):
    
    index_all = (torch.ones((307200,25)).cuda()*-1).int()
    
    intrinsic_path = f'scannetv2/processed/{seq_name}/intrinsic/intrinsic_depth.txt'
    intrinsics = np.loadtxt(intrinsic_path)
    intrinisc_cam_parameters = o3d.camera.PinholeCameraIntrinsic()
    intrinisc_cam_parameters.set_intrinsics(640, 480, intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2])
    
    pose_path = os.path.join(f'scannetv2/processed/{seq_name}/pose', str(frame_id) + '.txt')
    extrinsics = np.loadtxt(pose_path)
    
    if np.sum(np.isinf(extrinsics)) > 0:
        return None
    
    depth_path = os.path.join(f'scannetv2/processed/{seq_name}/depth', str(frame_id) + '.png')
    depth = cv2.imread(depth_path, -1)
    depth = depth / 1000.0
    depth = depth.astype(np.float32)
    
    depth_mask = get_depth_mask(depth)

    colored_pcld = backproject(depth, intrinisc_cam_parameters, extrinsics)
    view_points = torch.tensor(np.asarray(colored_pcld.points)).cuda().float()
    _, index_view, _ = ball_query(view_points.unsqueeze(0), scene_points.unsqueeze(0), K=25, radius=0.04, return_nn=False)
    index_all[depth_mask] = index_view[0].int()
    
    return index_all

def i2p(scene_points, frame_list, superpoint, seq_name):
    
    scene_points_num = len(scene_points)

    scene_points = torch.tensor(scene_points).float().cuda()
    feat_2d = torch.zeros((scene_points_num,1024)).cuda()
    feat_2d_num = torch.zeros((scene_points_num)).cuda()
     
    superpoint = torch.tensor(superpoint).long().cuda()
    
    for frame_id in tqdm(frame_list):
        
        #print(frame_id)
        
        i2p_index = turn_pixel_to_point(scene_points, frame_id, seq_name)
        assert scene_points.shape[0] == len(superpoint)
        if i2p_index == None:continue
        
        image = Image.open(os.path.join('scannetv2/processed',seq_name,'color',str(frame_id)+'.jpg'))
        image_input = clip_preprocess(image).unsqueeze(0).cuda()
        # Please modify the output of the original CLIP's visual encoder to obtain 'tokens'.
        # it is usually located in '<your python env dir>/site-packages/open_clip/transformer.py/<class VisionTransformer/forward function>'
        img_feats = clip_model.encode_image(image_input) 
        
        img_feats = img_feats.transpose(2,1).reshape(1,-1,24,24)
        img_feats = F.interpolate(img_feats, size=(480,640), mode="bilinear", align_corners=False)
        img_feats = img_feats.squeeze(0).reshape(1024,-1).transpose(0,1)
        
        for id in i2p_index.unique():
            if id == -1:continue
            mask = (i2p_index == id).sum(-1) > 0
            feat_2d_num[id] += mask.sum()
            feat_2d[id] += img_feats[mask].sum(0)
            
    feat_2d = feat_2d / feat_2d_num.unsqueeze(-1)
    feat_2d[torch.isnan(feat_2d)] = 0
    sp_feat_2d = scatter_mean(feat_2d, superpoint, dim=0)
    out = os.path.join('clip-feat',seq_name+'.pth')
    torch.save(sp_feat_2d.cpu(), out)

def main(seq_name, superpoint):
    mesh = o3d.io.read_point_cloud(f'scannetv2/processed/{seq_name}/{seq_name}_vh_clean_2.ply')
    scene_points = np.asarray(mesh.points)

    step = 10 # you can change this to 20 to speed up
    image_list = os.listdir(f'scannetv2/processed/{seq_name}/color')
    image_list = sorted(image_list, key=lambda x: int(x.split('.')[0]))
    end = int(image_list[-1].split('.')[0]) + 1
    frame_list = list(np.arange(0, end, step))

    with torch.no_grad():
        i2p(scene_points, frame_list, superpoint, seq_name)

if __name__ == '__main__':
    file_path = 'scannetv2/scannetv2.txt'
    with open(file_path, 'r') as f:
        seq_name_list = f.readlines()
    seq_name_list = [seq_name.strip() for seq_name in seq_name_list]
    output_path = './clip-feat'
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for seq_name in tqdm(seq_name_list):
        for split in ['val','train']:
            if os.path.exists("./scannetv2/"+split+'/'+seq_name+'_refer.pth'):
                _, _, superpoint, _, _ = torch.load("./scannetv2/"+split+'/'+seq_name+'_refer.pth')
        main(seq_name, superpoint)