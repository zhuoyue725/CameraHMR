import argparse
from mesh_estimator import HumanMeshEstimator


def make_parser():
    parser = argparse.ArgumentParser(description='CameraHMR Regressor')
    parser.add_argument("--image_folder", "--image_folder", type=str, 
        help="Path to input image folder.")
    parser.add_argument("--output_folder", "--output_folder", type=str,
        help="Path to folder output folder.")
    return parser

def main():

    parser = make_parser()
    args = parser.parse_args()
    estimator = HumanMeshEstimator()
    # estimator.run_on_images(args.image_folder, args.output_folder)
    # 3dpw中的对应图片

    idx_list = [11040, 1472, 16384, 17216, 17248, 18784, 18816, 18848, 19072, 19296, 19328, 20064, 20928, 21792, 21856, 22656, 31648, 31680, 3680, 6272, 7296, 12192, 12768, 1376, 13824, 13888, 1408, 14208, 15168, 15200, 15232, 15296, 15328, 24384, 24416, 24448, 24480, 24512, 24544, 24576, 24960, 25472, 34752, 35483, 640, 672, 704, 8864, 8896]
    # emdb结果
    # idx_list = [0, 10624, 10656, 10688, 10848, 11808, 11872, 12256, 12384, 12672, 13024, 13056, 13088, 13120, 13280, 13312, 13600, 13728, 14848, 15744, 15776, 15840, 15904, 16064, 1952, 1984, 21760, 2176, 22176, 22272, 3008, 384, 3904, 4576, 6112, 7456, 7488, 7520, 7552, 7584, 7616, 7648, 7680, 7712, 8064, 8576, 8768, 8800, 9280, 9536,1312, 2112, 3680, 4768,]
    #             ]
    # idx_list = [964]
    # 10240, 10816, 541, 632, 123, 714, 964
    for idx in idx_list:
        print(f"Starting processing from index {idx}.")
        # estimator.run_on_3dpw_images(idx, args.output_folder)
        # estimator.run_on_emdb_images(idx, args.output_folder)
        estimator.save_emdb_images(idx, args.output_folder)
        # estimator.save_3dpw_images(idx, args.output_folder)
        print(f"Finished processing from index {idx}.")
    
if __name__=='__main__':
    main()
# python demo.py --image_folder data/EMDB --output_folder output/emdb_depth > /home/zzb/pydata/recons/CameraHMR/output/emdb_depth/run.log 2>&1
# 解压：tar -xzvf P0.tar.gz
# 压缩：zip -r 3dpw_depth.zip /home/zzb/pydata/recons/CameraHMR/output/3dpw_depth/
    
    # 轮廓的权重/法向权重/2D关键点权重，设置的法向量权重比较大，所以可能出现轮廓没对齐但是法向量对齐