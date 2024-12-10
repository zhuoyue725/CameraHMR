CHECKPOINT_PATH='data/pretrained-models/camerahmr_checkpoint_cleaned.ckpt'
CAM_MODEL_CKPT='data/pretrained-models/cam_model_cleaned.ckpt'
SMPL_MEAN_PARAMS_FILE='data/smpl_mean_params.npz'
SMPL_MODEL_PATH='data/models/SMPL/SMPL_NEUTRAL.pkl'
DETECTRON_CKPT='data/pretrained-models/model_final_f05665.pkl'
DETECTRON_CFG='core/utils/cascade_mask_rcnn_vitdet_h_75ep.py'
TRANSFORMER_DECODER={'depth': 6,
                    'heads': 8,
                    'mlp_dim': 1024,
                    'dim_head': 64,
                    'dropout': 0.0,
                    'emb_dropout': 0.0,
                    'norm': 'layer',
                    'context_dim': 1280}

IMAGE_SIZE = 256
IMAGE_MEAN = [0.485, 0.456, 0.406]
IMAGE_STD = [0.229, 0.224, 0.225]
NUM_POSE_PARAMS = 23
NUM_BETAS = 10