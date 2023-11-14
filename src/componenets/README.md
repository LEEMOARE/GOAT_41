# Components


## 1. NIH Dataset
National Institutes of Health Chest X-Ray Dataset. [[paper]](https://openaccess.thecvf.com/content_cvpr_2017/papers/Wang_ChestX-ray8_Hospital-Scale_Chest_CVPR_2017_paper.pdf)[[download]](https://nihcc.app.box.com/v/ChestXray-NIHCC)

<br>
<div align="center">
<b>NIH Annotation</b>

custom NIH has below keys. 

</div>

```python
# example
{'age': 58,
 'filename': '00000001_000.png',
 'followup': 0,
 'gender': 'm',
 'id': 0,
 'label_indexes': [10],
 'label_names': ['cardiomegaly'],
 'pid': 1,
 'split': 'train',
 'view_position': 'pa'}
```

<br>


### 1.1 Data Distribution

In our dataset, we have several columns representing different aspects of the data. Here is a brief description of each column:

- Image Index : This column is a column representing the index or filename of the image and is used as a unique identifier for each image.
- Finding Labels : This column contains medical findings or disease labels found in the image. Multiple labels can be separated by commas.
- Follow-up #: This column represents the number of follow-ups of the patient. It can be used to connect multiple images of the same patient.
- Patient Age: This column represents the patient's age.
- Patient Gender: This column is a column representing the sex of the patient.
- View Position : This column is a column representing the patient's body part or location during image capture.
- Label Numbers : This column is a numeric column of medical findings about images.This number represents a label and can be represented as text in the "Finding Labels" column.


### 1.2 Gender Distribution

| class name | counts  | ratio |
| :--------: | :-----: | :---: |
|     M      | 63,340  |  57%  |
|     F      | 48,780  |  43%  |
|   total    | 112,120 | 100%  |

### 1.3 View Position Distribution

- Posteroanterior (PA) and Anteroposterior (AP) refer to the direction of the X-ray beam in chest imaging. The PA view is taken standing up and creates an image as the x-rays travel through the patient's back and toward the abdomen. AP view, unlike PA, is an imaging technique in which radiation first passes through the abdomen and then exits through the back. This photo was taken while lying down.

| Class Name | Counts  | Ratio |
| :--------: | :-----: | :---: |
|     PA     | 67,310  |  60%  |
|     AP     | 44,810  |  40%  |
|   total    | 112,120 | 100%  |

### 1.4 Age Distribution

전체 나이 분포 및 성별에 따른 나이 분포
| Category | mean  |   std   |
| :------: | :---: | :-----: |
|    M     | 47.19 | ± 17.23 |
|    F     | 46.52 | ± 16.32 |
|  total   | 46.90 | ± 16.84 |


| Age Group | Counts | Ratio  |
| :-------: | :----: | :----: |
|    0-9    |  1403  | 0.012  |
|   10-19   |  5421  | 0.048  |
|   20-29   | 12798  | 0.114  |
|   30-39   | 16313  | 0.145  |
|   40-49   | 21731  | 0.193  |
|   50-59   | 27406  | 0.244  |
|   60-69   | 19272  | 0.171  |
|   70-79   |  6641  | 0.059  |
|   80-89   |  1054  | 0.009  |
|   90-99   |   65   | 0.0005 |
|   100+    |   16   | 0.0001 |

### 1.5 Label Numbering

In our dataset, we have converted the labels representing different lesions into numbers for easier processing and analysis. Here is the mapping we used:

|    Lesion Name     | Number |                       Description                       | Count | Ratio  |
| :----------------: | :----: | :-----------------------------------------------------: | :---: | :----: |
|      Effusion      |   1    |                 Fluid around the lungs                  | 13317 | 0.094  |
|     Emphysema      |   2    |           Damage to the air sacs in the lungs           | 2516  | 0.017  |
|    Atelectasis     |   3    |                Collapsed or closed lung                 | 11559 | 0.081  |
|       Edema        |   4    |             Swelling caused by excess fluid             | 2303  | 0.016  |
|   Consolidation    |   5    |           Area of the lung filled with fluid            | 4667  | 0.032  |
| Pleural_Thickening |   6    |          Thickening of the lining of the lungs          | 3385  | 0.023  |
|       Hernia       |   7    | Organ pushes through an opening in the muscle or tissue |  227  | 0.0016 |
|        Mass        |   8    |            Abnormal lump or growth of cells             | 5782  | 0.040  |
|     No Finding     |   9    |                  No abnormality found                   | 60361 | 0.426  |
|    Cardiomegaly    |   10   |                     Enlarged heart                      | 2776  | 0.019  |
|       Nodule       |   11   |                   Small lump of cells                   | 6331  | 0.044  |
|    Pneumothorax    |   12   |                     Collapsed lung                      | 5302  | 0.037  |
|     Pneumonia      |   13   |                Inflammation of the lungs                | 1431  | 0.010  |
|      Fibrosis      |   14   |                  Scarring of the lungs                  | 1686  | 0.011  |
|    Infiltration    |   15   |              Substance passes into tissue               | 19894 | 0.140  |

You can refer to this table to understand the correspondence between the lesion names and the numbers used in the analysis.

 
