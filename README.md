# SCAP: A Novel Lightweight Model for Efficient Image Captioning through Enhanced Feature Sifting

## Project Overview

* This model utilizes the MSCOCO-2014 dataset for training and testing, aiming to develop image captioning capabilities by learning from (image, text) pairs, enabling the generation of descriptive text that accurately corresponds to input images.
* The model is based on an improved transformer architecture, reusing the dataset construction classes from `dataset/` and evaluation metric classes from `evaluation/`.
* SCAP is a novel lightweight model that enhances image captioning through a unique sifting attention mechanism. SCAP introduces summary and forget modules within its encoder to refine visual information, discarding noise and retaining essential details. The hierarchical decoder then leverages sifting attention to align image features with text captions, generating accurate and contextually relevant descriptions.

## Contributions

* An original end-to-end model, SCAP, is proposed to align visual features with linguistic features, thereby effectively and accurately implementing image captioning.
* We introduce an enhanced attention mechanism model that, by combining summary module and forget module, achieves a rational mapping of multiple modelities of information, improving the efficiency of modelity fusion.
* SCAP adopts a hierarchical induction decoder, achieving layer-wise propagation of modelity information and preventing mismatch of information intensity during information fusion.
* We conduct extensive experiments demonstrating that our model achieves high performance under the premise of being lightweight, outperforming models of similar parameter scale.

## Usage

* Install the `conda` environment using `environment.yml`. This project uses Python 3.6 and PyTorch 1.1.0. Note that an incorrect `cudnn` version may lead to errors.

* Download the `coco_detections.hdf5` file

  ```
  Annotations download link: https://pan.baidu.com/s/1zDrue0kgWapxNjaItS9PKg?pwd=3b1e 
  Extraction code: 3b1e 
  ```

* Training

  ```python
  python train.py --batch_size 50 --head 8 --warmup 10000 --features_path {path_to_coco_detections} --annotation_folder {path_to_annotations_folder} --workers 90
  ```

* Testing

  ```python
  python test.py
  ```

