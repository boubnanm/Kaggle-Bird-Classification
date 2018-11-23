
## Inclass Kaggle competition for Bird Classification

### Bird Classification
This Kaggle competition, with the rest of the Object recognition and computer vision MVA Class 2018/2019, was on a subset of the Caltech-UCSD Birds-200-2011 bird dataset.

Link : [https://www.kaggle.com/c/mva-recvis-2018](https://www.kaggle.com/c/mva-recvis-2018)

#### Requirements
1. Install PyTorch from http://pytorch.org

2. Run the following command to install additional dependencies

```bash
pip install -r requirements.txt
```

#### Dataset
We will be using a dataset containing 200 different classes of birds adapted from the [CUB-200-2011 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).
The training/validation/test images used for this model can be downloaded from [here](https://www.di.ens.fr/willow/teaching/recvis18/assignment3/bird_dataset.zip). The test image labels are not provided.

#### Training and validating the model
For the model, we used two pretrained models (ResNet152 and InceptionV3) to extract two features vectors. A classifier is added to classify the images using the stacked extracted features. See `model.py` and the attached paper for more details.

To train the model with default parameters, run the following command :

```bash
python main.py
```

By default :
- The images are loaded and resized to 331x331 pixels and normalized to zero-mean and standard deviation of 1. See data.py for the `data_train_transforms`. In order to preserve the ratio of the images, a padding option is available by running the following command (this option is **disabled** by default) :

```bash
python main.py --pad
```

- The data is augmented by preprocessing the images using YoloV3 to detect birds and add cropped images centered on the birds. The outputed images are saved at `bird_dataset_output`. See `YOLO_model.py` for the  [YoloV3](https://github.com/eriklindernoren/PyTorch-YOLOv3) code.
To deactivate the detection process and train on the original training and test sets, run the following command :

```bash
python main.py --no-crop
```

An other option for training the model on the training and validation sets is available by running the following command (this option is **disabled** by default): 

```bash
python main.py --merge
```

See attached paper for default training parameters.

#### Evaluating the model on the test set

As the model trains, model checkpoints are saved to files such as `model_x.pth` to the current working directory.
You can take one of the checkpoints and run:

```
python evaluate.py --data [data_dir] --model [model_file]
```

That generates a file `kaggle.csv` that you can upload to the private kaggle competition website.

By default, the cropped images (bird_dataset_ouput) are used as the default directory.


#### Acknowledgments
Adapted from Rob Fergus and Soumith Chintala https://github.com/soumith/traffic-sign-detection-homework for the third assignment of the Object recognition and computer vision MVA Class 2018/2019, taught by [J. Ponce](https://www.di.ens.fr/~ponce/), [I. Laptev](https://www.di.ens.fr/~laptev/), [C. Schmid](http://lear.inrialpes.fr/~schmid/), and [J. Sivic](https://www.di.ens.fr/~josef/)