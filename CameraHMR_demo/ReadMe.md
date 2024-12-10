
<div align="center">

# **CameraHMR: Aligning People with Perspective**  
## **3DV 2025**  
🔗 [**Project Page**](https://camerahmr.is.tue.mpg.de) | 📄 [**ArXiV Paper**](https://arxiv.org/abs/2411.08128) | 🎥 [**Video**](https://youtu.be/aclZSzUIj5o)

</div>

---

## 🚀 **Release**

- ** CameraHMR Demo Code**



## 🕒 **Coming Soon**

-  **CameraHMR Training Code**  
-  **CamSMPLify Code**  
-  **HumanFoV and DenseKP Code**


##  **Data**

You can download the fitted SMPL parameters for **INSTA/AIC images** via the [CameraHMR website](https://camerahmr.is.tue.mpg.de/index.html) (registration required).

Alternatively, use the following script:

```bash
bash fetch_training_data.sh
```

> **Note:** We cannot provide the original AIC/INSTA images. These images must be obtained from their original sources. For convenience, you can use the [4D-Humans repository](https://github.com/shubham-goel/4D-Humans?tab=readme-ov-file), which offers these images in WebDataset format. To extract images from the WebDataset, refer to [this script](core/utils/extract_images_from4dhumans.py).

###  **Visualize the Fitted SMPL Mesh**

To overlay the fitted SMPL mesh on your images, use the following command:

```bash
python dataset_vis.py --image_folder path_to_img_folder --output_folder path_for_output_file --npz_path path_to_npz_file
```
path_to_img_folder corresponds to path of download INSTA/AIC images. path_to_npz_file corresponds to downloaded SMPL params.



## 🎬 **Demo**

###  **Download Required Files**

Download necessary demo files using:

```bash
bash fetch_demo_data.sh
```

Alternatively, download files manually from the [CameraHMR website](https://camerahmr.is.tue.mpg.de). Ensure to update paths in [`constants.py`](core/constants.py) if doing so manually.

###  **Generate 3D SMPL Mesh and Overlay Image**

Run the demo with:

```bash
python demo.py --image_folder path_to_input_images --output_folder path_to_output_folder
```

## 🙌 **Acknowledgements**

This project leverages outstanding resources from:

- [ 4D-Humans](https://github.com/shubham-goel/4D-Humans?tab=readme-ov-file)  
- [ BEDLAM](https://bedlam.is.tue.mpg.de/)  
- [ SMPLify](https://smplify.is.tue.mpg.de/)  
- [ ViTPose](https://github.com/ViTAE-Transformer/ViTPose)  
- [ Detectron2](https://github.com/facebookresearch/detectron2)


## 📚 **Citation**

If you find **CameraHMR** useful in your work, please cite:

```bibtex
@article{patel2024camerahmr,
  title={CameraHMR: Aligning People with Perspective},
  author={Patel, Priyanka and Black, Michael J},
  journal={arXiv preprint arXiv:2411.08128},
  year={2024}
}
```



<div align="center">

✨ **Thank you for your interest in CameraHMR!** ✨

</div>

