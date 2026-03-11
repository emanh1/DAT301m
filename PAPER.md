# SSMD: Semi-Supervised Medical Image Detection with Adaptive Consistency and Heterogeneous Perturbation

![Diagram illustrating the Semi-Supervised Medical image Detector (SSMD) architecture. It shows a Student Detector (green) and a Teacher Detector (blue) processing inputs from a 'Pool of Pert.' (purple dots). The Student Detector uses 'Noisy Residual Blocks' (N) and a 'Supervised Loss' (S). The Teacher Detector uses 'Adaptive Consistency Cost' (C) and 'Supervised Loss' (S). The 'Pool of Pert.' includes 'Adv.' (adversarial perturbation) and is connected to both detectors. Red boxes indicate regions of interest or perturbations.](68ac34ff111db52afaa786afcb8346c3_img.jpg)

The diagram illustrates the Semi-Supervised Medical image Detector (SSMD) architecture. It features a Student Detector (green) and a Teacher Detector (blue). The Student Detector processes inputs through a series of Noisy Residual Blocks (N) and calculates a Supervised Loss (S). The Teacher Detector also processes inputs through a series of blocks and calculates an Adaptive Consistency Cost (C) and a Supervised Loss (S). A 'Pool of Pert.' (purple dots) provides perturbations to both detectors. The 'Adv.' (adversarial perturbation) is specifically applied to the input of the Teacher Detector. Red boxes indicate regions of interest or perturbations on the feature maps.

Figure 1: Overview of the proposed Semi-Supervised Medical image Detector (SSMD). Two feature pyramid networks are utilized to predict consistent outputs. **Pool of Pert.** refers to a set of perturbation strategies, which include horizontal flip, vertical flip, random rotation and adversarial perturbation (denoted as **Adv.**). Note that the adversarial perturbation is only applied to the input of teacher network.

In this paper, we mainly focus on two object detection tasks:

lesions detection and nuclei detection. Locating lesions and abnormalities in CT scans is a primary object detection task for radiologists. They need to find out the location of lesions, and describe the related attributes in radiological reports. An automatic lesion detector could not only reduce the workload of radiologists, but also benefit the areas that have a shortage of experienced radiologists. DeepLesion, a representative lesion detection dataset, contains 32,120 CT slices and 32,735 annotated lesions. Another fundamental object detection task in medical image analysis is nuclei or cell detection, which helps measure quantitative information to better understand disease progression. 2018 Data Science Bowl introduced a nuclei detection dataset that consists of about 27,000 cells. To reduce the high cost of medical image annotations, it is important to train a robust detector with not only labeled but also unlabeled medical images. Thus, in this paper we propose a novel Semi-Supervised Medical image Detector (SSMD) that can make use of unlabeled medical images to produce robust representations in an effective way.

Our proposed semi-supervised detector addresses two problems when applying the consistency regularization to medical image detection: a) too many background proposals may dominate the training procedure and b) mediocre augmentation strategies (such as horizontal flip and translation) cannot well regularize visual representations. For issue a), we propose *adaptive consistency cost* to adaptively scale the loss values, where the scaling factor decays to zero as confidence of the background class increases. Intuitively, the proposed adaptive mechanism can automatically down-weight the influences of background proposals. As for problem b), we introduce a set of heterogeneous perturbation methods to advance the regularization effect of the consistency loss. The core idea behind is that we want the detector to capture the invariant representations as we apply

various perturbations. We believe these representations are more robust and thus more generalized, even under various real-world noise.

![Diagram illustrating the general network architecture for medical image detection, comparing one-stage and two-stage detectors. The One-stage Detector (blue dashed box) takes an input image, passes it through a Backbone network, and then through Box Head1 to produce Proposal Boxes. The Two-stage Detector (green dashed box) takes the input image through the Backbone, then through Box Head1 to produce Proposal Boxes, which are then refined into RoI Boxes. These RoI Boxes are then passed through Box Head2 (which includes Box Category Classifier2 and Box Location Regressor2) to produce the final output. A green arrow labeled 'RoI Pooling' shows the flow from the RoI Boxes to Box Head2.](547f726730e589392f239257a833ede3_img.jpg)

Figure 2: General network architecture for medical image detection. RoI is an abbreviation for region of interest.

## 3 Method

Existing deep learning based medical image detection methods are usually anchor-based, which predict the relative position and scale factors between each object box and some pre-defined anchor boxes. These methods could be divided into two types: one-stage [23, 16, 48] and two-stage detectors, as shown in Figure 2. Given a one-stage detector, it first produces a large number of proposal boxes via the backbone network, after which one category classifier and

one location regressor are employed to deal with these boxes. In contrast, the two-stage pipeline requires one more box head which is responsible for refining those RoI boxes produced by the first box head. During the training stage, the overall loss function can be summarized as:

$$\text{CE}(p^c, p_{gt}^c) + \text{SmoothL1}(p_s^{\{x,y,w,h\}}, p_{gt}^{\{x,y,w,h\}}), \quad (1)$$

where  $p^c$  stands for the class prediction,  $p_{gt}^c$  denotes the ground truth class. Similarly,  $p^{\{x,y,w,h\}}$  stands for coordinate and box size predictions while  $p_{gt}^{\{x,y,w,h\}}$  represents their ground truth targets. CE stands for the cross entropy loss to train box classifiers, and SmoothL1 represents the smooth L1 loss which is employed to train box regressors. Recently, focal loss is often used to replace the cross entropy loss when the number of proposals is extremely large [37, 48]. Both one-stage and two-stage approaches require backbone networks to extract image features, where 2D and 3D deep neural networks are used according to the types of input data. Specifically, for 2D tasks, VGG-16 and ResNet are two widely adopted architectures. For 3D tasks, 3D ResNet [37, 48] and 3D U-Net are two representatives.

In this paper, we propose SSMD which incorporates medical image detection with semi-supervised learning. Compared to semi-supervised classification/segmentation, SSMD focuses more on instance regions instead of the whole image in classification or individual pixels in segmentation. Accordingly, to better regularize instance regions in detection, our SSMD addresses the importance of adding consistency to instance locations which are usually ignored in semi-supervised classification/segmentation.

In the following we describe three major contributions of SSMD: the adaptive consistency cost function, the noisy residual block and the instance-level adversarial perturbation strategy. We provide an overview in Fig. 1 in which a student-teacher framework is employed to generate predictions for shared inputs with different perturbation strategies. For labeled images, the proposed method uses an adaptive consistency cost and the supervised loss. For unlabeled data, only the adaptive consistency cost is used. The consistency loss is calculated with the predicted proposals at each spatial position and each scale.

To make use of unlabeled images, it is necessary to mine the data to generate intrinsic supervision signals which can be further incorporated into the training process. Nowadays, most semi-supervised deep learning approaches focused on improving image classification results by keeping consistency within perturbed pairs. They require paired inputs where each pair contains the same image with different perturbation strategies. After feeding these pairs to neural networks, semi-supervised approaches force the outputs of each pair to be as close as possible. The most common perturbation methods can be summarized as: translation, rotation and horizontal flip. In this paper, we propose three more perturbation approaches: noisy residual block in feature space, instance-level adversarial perturbation and cutout in image space.

![Diagram of the proposed noisy residual block. The input is a feature map X^l in R^{C x H x W}. It undergoes channel-wise average pooling to produce X^0 in R^{C x 1 x 1}. X^0 is then processed by a 1 x 1 convolution to produce X^p in R^{C x 1 x 1}. X^p is passed through a sigmoid function. Simultaneously, X^l is perturbed with Gaussian noise to produce X^n in R^{C x H x W}. The sigmoid output is element-wise multiplied (⊗) with X^n. The result is then added (⊕) to the original input X^l to produce the final output X^q in R^{C x H x W}.](3121ebddccf183ca63bb9781be440a7e_img.jpg)

Figure 3: Illustration of the proposed noisy residual block. The proposed module adds noise perturbations to a residual block. Note that different colors mean different channels.  $\otimes$  stands for channel-wise multiplication while  $\oplus$  represents channel-wise addition.

### 3.1 Adaptive Consistency Cost

As shown in Fig 1, the proposed SSMD model contains a student detector and a teacher detector where each network contains a feature pyramid network. We adopt a parameter sharing approach proposed by where the teacher model uses the exponential moving average (EMA) weights of the student model. We denote the weights of the teacher model and the student model as  $\theta_t$  and  $\theta_s$  respectively.  $\theta_t^n$  denotes the weights of the teacher network at training step  $n$  and is updated as follows:

$$\theta_t^n = \alpha\theta_t^{n-1} + (1 - \alpha)\theta_s^n \quad (2)$$

where both  $\theta_0^t$  and  $\theta_0^s$  are independently initialized. During the training stage, for the student branch we apply random rotation and then randomly mask out some rectangle regions, which is known as cutout. As for the teacher branch, we first apply horizontal flip and cutout to the augmented input of the student branch, and then add instance-level adversarial perturbation to it. Different from CSD, we propose to utilize an adaptive version of consistency cost to exploit unlabeled images and synthesize adversarial samples. The detector is based on RetinaNet which is to predict

the positions of proposals relative to pre-defined anchors.  $p^x, p^y, p^w, p^h$ , which are outputs of the proposed detector, denote four scale factors:

$$\begin{aligned} p^x &= (x - x^a)/w^a, \quad p^y = (y - y^a)/h^a, \\ p^w &= \log(w/w^a), \quad p^h = \log(h/h^a), \end{aligned} \tag{3}$$

where  $x, y$  are the coordinate of a proposal center.  $w$  and  $h$  represent the width and the height of a proposal.  $x^a, y^a, w^a$  and  $h^a$  are variables for default anchors. Let  $p^c$  denote the predicted probability distribution of different categories (after softmax). The whole procedure of the proposed semi-supervised medical detection is provided in Algorithm 1. We first apply different perturbations to a batch of labeled images  $\mathcal{X}$  for two branches, respectively. After the forward pass, we obtain the predictions of classes and box coordinates. For each labeled image in  $\mathcal{X}$ , its supervised loss ( $\text{loss}_{sup}$ , shown in Line 8 of Algorithm 1), which consists of a cross entropy loss (CE) and a smooth L1 loss (SmoothL1), can be directly calculated between the prediction and the ground truth.

To regularize the final predictions between the labeled images  $\mathcal{X}$  and the unlabeled images  $\tilde{\mathcal{X}}$ , we apply a consistency cost which includes a KL-Divergence loss (KL) and a mean squared error (MSE) loss, as shown in Line 11 of Algorithm 1. Assume that the output distributions of the teacher and the student models are close. Then KL loss is adopted for classification consistency, to measure the output difference between the teacher and the student networks. For location consistency, we follow the setting of CSD and adopt MSE loss. Specifically, our proposed adaptive cost function contains a dynamic instance weight  $\mathbf{W}(p_s^c, p_t^c)$  which is defined as:

$$\mathbf{W}(p_s^c, p_t^c) = \frac{(1 - p_s^c[0])^2 + (1 - p_t^c[0])^2}{2}, \tag{4}$$

where  $p_s^c[0]$  refers to the probability belonging to the background category, predicted by the student network.  $p_t^c[0]$  shares the same definition and is predicted by the teacher network. In our implementation, we treat the features of different levels equally in the adaptive consistency cost. For each feature level, the weight of the adaptive cost is equal to 1. The MSE loss displayed in Line 11 and Line 12 of Algorithm 1 is computed as:

$$\begin{aligned} \text{MSE}(p_s^{\{x,y,w,h\}}, p_t^{\{x,y,w,h\}}) &= \text{MSE}(p_s^x, p_t^x) + \text{MSE}(p_s^y, p_t^y) \\ &\quad + \text{MSE}(p_s^w, p_t^w) + \text{MSE}(p_s^h, p_t^h), \end{aligned} \tag{5}$$

where  $p_s$  and  $p_t$  are the predictions of student model and teacher model, respectively. For the prediction of unlabeled data  $\tilde{p}$ , we calculate its MSE loss  $\text{MSE}(\tilde{p}_s^{\{x,y,w,h\}}, \tilde{p}_t^{\{x,y,w,h\}})$  in a similar

way with Equation (5). Note that during the inference stage, only the student network  $f_{\theta_s^N}(\cdot)$  is used to perform final predictions.

The proposed adaptive consistency cost takes into account the predicted confidence of proposals at each spatial position. Given a proposal with high foreground probability, it would result in a higher weight of the consistency cost than those of easily recognized background samples. This mechanism helps the model apply more regularization effects to objects instead of the meaningless background. In practice, this adaptive cost is applicable to both labeled and unlabeled medical images, making proposed detector more effective in the setting of small amounts of labeled data.

### 3.2 Noisy Residual Block

In this part we propose *noisy residual block* that adds noise to intermediate feature maps. The proposed noisy residual block can be regarded as a perturbation strategy working in a feature space. As shown in Fig.3, we modify the classical residual block and append an attention-based mechanism. We name the proposed module noisy residual block, since it introduces noise perturbations to a residual block. More details are in the following.

The input to layer  $l$  is denoted as  $X^l \in \mathcal{R}^{C \times H \times W}$ . The proposed noisy residual block first applies a channel-wise average pooling to  $X^l$  and then adopts a  $1 \times 1$  convolutional operation:

$$X^p = \text{conv}(\text{AvgPool}(X^l)). \quad (6)$$

where  $X^p \in \mathcal{R}^{C \times 1 \times 1}$  and *AvgPool* is the abbreviation of global average pooling. For each layer  $l$ , we sample a Gaussian noise map  $X^n \in \mathcal{R}^{C \times H \times W}$  where each component is drawn from a Gaussian distribution  $\mathcal{N}(\mu, \sigma)$ .  $\mu$  and  $\sigma$  stand for the mean and standard deviation, respectively. Meanwhile, we employ a scaled sigmoid function to normalize  $X^p$ . A channel-wise multiplication is performed between  $X^p$  and  $X^n$ . Finally,  $X^q$  can be computed by adding the multiplication result to the input feature  $X^l$ :

$$X^q = (X^n \otimes \text{sigmoid}(\gamma X^p)) \oplus X^l. \quad (7)$$

where  $\gamma$  is a scale factor. Here we employ a *sigmoid* function to adaptively control the noise level of different channels in the noise perturbation.  $X^q$  serves as the output of the noisy residual block and will be passed to following layers.

An intuitive understanding of the noisy residual block is to add "appropriate" noise to intermediate representations. For example, shallow layers are supposed to have wild noise as they are foundations of the whole network. We believe the degree of the embedded noise should be determined and can be learned by the representations themselves. Motivated by this idea, the noisy residual block learns channel-wise attentions to apply channel-dependent noise to feature maps. Moreover, we employ a residual connection to maintain the stability of the training process.

### 3.3 Instance-level Adversarial Perturbation based on Consistency Regularization

Adversarial training has been widely adopted as a useful way to improve semi-supervised classification and segmentation. In contrast, the detection problem focuses more on instances instead of pixels in classification or segmentation. Thus, the methods designed for classification/segmentation may not be suitable for detection because they treat all pixels equally. In this section, we propose an instance-level adversarial perturbation strategy to address this issue.

Let  $r_{adv}$  denote the adversarial perturbations added to the input image. In each training iteration,  $r_{adv}$  is first initialized from a normalized Gaussian distribution and has the same shape as  $\mathcal{X}$  and  $\tilde{\mathcal{X}}$ . Then, a scaled  $r_{adv}$  is added to the original image as:

$$\text{Adv.}(\mathcal{X}) = \mathcal{X} + \xi r_{adv}, \quad (8)$$

where  $\xi$  is a scale factor satisfying  $0 < \xi \le 1$ . Classical adversarial examples work by causing classifiers to predict a wrong category. However, in SSMD, the goal of adding adversarial perturbations is to increase the difficulty of performing consistency regularization. Note that similar computation process can also be applied to  $\tilde{\mathcal{X}}$ .

We pass  $\{\mathcal{X}, \tilde{\mathcal{X}}\}$  and  $\{\text{Adv.}(\mathcal{X}), \text{Adv.}(\tilde{\mathcal{X}})\}$  to student and teacher networks respectively, to obtain the consistency loss  $\text{loss}_{cont}$  (shown in Line 11 of Algorithm 1). Only the high-confidence predictions are used to compute the consistency loss for gradient backward when applying adversarial perturbation. The gradient  $g$  and the adversarial perturbations  $r_{adv}$  are computed as:

$$g = \nabla_{r_{adv}} \text{loss}_{cont} * \mathbf{1}[\sum p_s^c \| \sum \tilde{p}_s^c > \tau], \quad (9)$$

$$r_{adv} = \epsilon \frac{g}{\|g\|}.$$

where the symbol  $\sum$  denotes the sum of all foreground classes.  $\mathbb{1}[\cdot]$  is an indicator function which equals 1 when  $\sum p_s^c$  or  $\sum \tilde{p}_s^c$  is larger than a given threshold  $\tau$ .  $\epsilon$  is the strength of perturbation, controlling the magnitude of  $r_{adv}$ .  $\|\cdot\|$  stands for L2 normalization. After computing Equation (9),  $r_{adv}$  is added to  $\mathcal{X}$  to obtain the final perturbed input. In general, it requires an additional forward and backward pass to synthesize the perturbed input image before we feed these final inputs to the detection network. Such process is to maximize the effect of  $r_{adv}$  on  $\text{loss}_{cont}$ , and can be viewed as an adversarial process.

Similar to the adaptive cost, we design instance-level perturbation to amplify the influences of high-confidence foreground proposals while reducing the impacts of low-confidence ones. In practice, foreground pixels receive heavy adversarial noise while the perturbation of background pixels has much smaller magnitude. Such implementation makes the consistency loss focus more on foreground objects, producing effectively perturbed inputs.

## 4 Experiments

In this section, we first conduct ablation studies to better understand the strengths of different modules in the proposed method SSMD. Moreover, we design comprehensive experiments to verify the effectiveness of SSMD on various settings.

### 4.1 Dataset

The experiments are conducted on a nuclei dataset and a lesion database. For both datasets, we manually and randomly split the training set into labeled data and unlabeled data with fixed ratios in order to fit the setting of semi-supervised learning.

**Nuclei Dataset** In our experiments, we adopt the nuclei dataset introduced by 2018 Data Science Bowl (DSB, hosted by Kaggle). The dataset was acquired under a variety of conditions and includes nuclei images of different cell types, magnifications, and imaging modalities. The training set contains 522 nuclei images (80%) while the validation set has about 60 images (10%). The rest images are used for testing. On average, each image contains about 45 cells which are enough to train a robust nuclei detector. In practice, we only assign labels to some training images and take the other training images as unlabeled data. The evaluation metric is mAP

**DeepLesion Dataset** We also present experimental results on DeepLesion which is a large-scale public dataset containing 32,120 axial Computed Tomography (CT) slices of 10,594 studies collected from 4,427 patients. The dataset has 32,735 annotated lesion instances in total. Each slice contains 1~3 lesions. The additional slices above and beneath a target slice are regarded as relevant contexts of the target slice. These additional slices are of 30 mm. In most cases, a slice is 1 or 5 mm thick. The dataset covers a wide scope of lesions from lung, liver, mediastinum (essentially lymph hubs), kidney, pelvis, bone, midsection and delicate tissue. We test our proposed method on official testing set (15%) and report the sensitivity at 4 false positives (FPs). We directly use the training and validation set officially provided by DeepLesion.

### 4.2 Implementation Details

For DSB dataset, the proposed detector is built on top of an ImageNet-pretrained ResNet-50 which has five scales. Nine default anchors are adopted in each scale. The size of input images is  $448 \times 448$ . The batch size is 8. All models are trained for 100 epochs. Adam is utilized as the default optimizer with  $1e-5$  as the initial learning rate, which is then divided by 10 at the 75th ( $100 \times \frac{3}{4}$ ) epochs. For the supervised baseline, image rotation and horizontal image flipping are considered as default augmentation strategies. It is worth noting that the hyperparameter  $\lambda$  of consistency loss (shown in Line 13 of Algorithm 1) plays an important role during the training stage. We first gradually increase the value of  $\lambda$  to 1 in the first quarter of the training, and then decrease it to 0 in the last quarter. The formal definition of  $\lambda$  is:

$$\lambda = \begin{cases} e^{-5(1-\frac{4j}{N})^2}, & 0 \le j < \frac{N}{4} \\ 1, & \frac{N}{4} \le j < \frac{3N}{4} \\ e^{-12.5(1-\frac{7(N-j)}{N})^2}, & \frac{3N}{4} \le j \le N \end{cases} \quad (10)$$

where  $N$  is the number of training iterations and  $j$  is the iteration index. For DeepLesion dataset we simply follow the preprocessing method in [48] to resize each slice into  $512 \times 512$  pixels whose mean voxel-spacing is 0.802mm. We first clip the Hounsfield units (HU) to  $[-1100, 1100]$  and then normalize them to  $[-1, 1]$ . We compute the mean and standard deviation of the whole training set and use them to further normalize input slices. For both datasets, we set  $\gamma$  to 0.9 and the degree of random rotation is set to 10 degrees.

## --- **Algorithm 1** Procedure of Semi-Supervised Medical Detection ---

#### **Input:**

A batch of labeled images  $\mathcal{X} = \{x^1, \dots, x^B\}$ . A batch of unlabeled images  $\tilde{\mathcal{X}} = \{\tilde{x}^1, \dots, \tilde{x}^B\}$ .

$\backslash\backslash$   $B$  is the batch size

$f_{\theta_s^n}(\cdot)$ : Student network at time step  $n$ .  $f_{\theta_t^n}(\cdot)$ : Teacher network at time step  $n$ .

- 1:  $\mathcal{X}_s = \text{cutout}(\text{Rot.}(\mathcal{X})); \tilde{\mathcal{X}}_s = \text{cutout}(\text{Rot.}(\tilde{\mathcal{X}})) \backslash\backslash$  Augmentation for the student network
  - 2:  $\mathcal{X}_t = \text{Adv.}(\text{cutout}(\text{Rot.}(\text{Flip}(\mathcal{X}))))$ ;  $\tilde{\mathcal{X}}_t = \text{Adv.}(\text{cutout}(\text{Rot.}(\text{Flip}(\tilde{\mathcal{X}})))) \backslash\backslash$  Augmentation for the teacher network
  - 3:  $\backslash\backslash$  Forward pass to get predictions
  - 4: **for**  $n = 1$  to  $N$  **do**  $\backslash\backslash$   $N$  is number of training iterations
  - 5:  $\theta_t^n = \alpha\theta_t^{n-1} + (1 - \alpha)\theta_s^n \backslash\backslash$  Update teacher network
  - 6:  $\text{loss}_{\text{cont}} = 0$ ;  $\text{loss}_{\text{sup}} = 0 \backslash\backslash$  Initialize loss values
  - 7: **for**  $i = 1$  to  $B$  **do**
  - 8:  $p_s^c, p_s^x, p_s^y, p_s^w, p_s^h = f_{\theta_s^n}(x_s^i); \tilde{p}_s^c, \tilde{p}_s^x, \tilde{p}_s^y, \tilde{p}_s^w, \tilde{p}_s^h = f_{\theta_s^n}(\tilde{x}_s^i) \backslash\backslash$  Forward through student network
  - 9:  $p_t^c, p_t^x, p_t^y, p_t^w, p_t^h = f_{\theta_t^n}(x_t^i); \tilde{p}_t^c, \tilde{p}_t^x, \tilde{p}_t^y, \tilde{p}_t^w, \tilde{p}_t^h = f_{\theta_t^n}(\tilde{x}_t^i) \backslash\backslash$  Forward through teacher network
  - 10:  $\text{loss}_{\text{sup}} += \text{CE}(p_s^c, p_{gt}^c) + \text{SmoothL1}(p_s^{\{x,y,w,h\}}, p_{gt}^{\{x,y,w,h\}}) \backslash\backslash$  Apply supervised loss
  - 11:  $\text{loss}_{\text{cont}} += \mathbf{W}(p_s^c, p_t^c) \otimes (\text{KL}(p_s^c, p_t^c) + \text{MSE}(p_s^{\{x,y,w,h\}}, p_t^{\{x,y,w,h\}})) \backslash\backslash$  Consistency loss on labeled images
  - 12:  $\text{loss}_{\text{cont}} += \mathbf{W}(\tilde{p}_s^c, \tilde{p}_t^c) \otimes (\text{KL}(\tilde{p}_s^c, \tilde{p}_t^c) + \text{MSE}(\tilde{p}_s^{\{x,y,w,h\}}, \tilde{p}_t^{\{x,y,w,h\}})) \backslash\backslash$  Consistency loss on unlabeled images
  - 13:  $\text{loss} = \text{loss}_{\text{sup}} + \lambda \text{loss}_{\text{cont}} \backslash\backslash$   $\lambda$  is a hyperparameter
  - 14: **Backward(loss)**  $\backslash\backslash$  Update student network