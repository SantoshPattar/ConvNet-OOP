# Configuration Parameters Description

This file describes the JSON configuration parameters in Key:Value pair, along with their possible values for a given experiment.
Each parameter is described in the following syntax:

```
**Parameter_Name:**
	- Parameter_Description.
	- Possible_Values.
```
---

- **exp_name:**
	- Name of the experiment.
	- Each experiment will have only one value, *i.e* the name of the experiment to be carried out.

- **image_width:**
 	- Specifies the input image's width.
	- Value depends on the input dataset.

- **image_height:**
	- Specifies the input image's height.
	- Value depends on the input dataset.

- **image_channel:**
	- Specifies the channel/depth size of the input image.
	- Value depends on the type of the input image.

- **image_width:**
	- Specifies the size of the pixel for a given input image.
	- It's vaue depends on the type of the input dataset. 

- ***save_plots:**
	- Indicates whether to save or not the validation graphs for a given experiment.
	- true or false.

- **num_classes:**
	- Indicates the number of classes in the given dataset.
	- Value depends on the dataset of the experiment.

- **model_type:**
	- Shows the type of ConvNet model to be implemented.
	- Can be either Sequential or Functional.

- **no_of_layers:**
	- Represents the number of layers used in the ConvNet model.
	- An integer indicating the number of layers.

- **kernel_row:**
	- Mentions the size of a row for a given filter that will stride on the input data. 
	- Value depends on the selected filter size.

- **kernel_column:**
	- Mentions the column size of the filter image that will stride on the input data. 
	- Value depends on the selected filter size.

- **cmap_val:**
	- Enumerates the color of each channel in the input image.
	- Value depends on the type of images present in the dataset of the experiment. Commonly used values are RGB or Gray.

- **stride_size:**
	- Denotes the size of the stride for the feature learning.
	- stride_size must be an integer i.e., It should be in the range [greater than zero but less than input image_size] and also it must satisfy the stride validation formula.

- **padding:**
	- Establishes the type of padding performed on the border of the input data.
	- Same or Valid.

- **padding_size:**
	- Indicates the number of row and column to be padded on the border of the input data.
	- must be >= 0.

- **no_of_filters_l1:**
	- Indicates the number of filters to be included in the Conv layer 1 of the ConvNet model. 
	- no_of_filters > 0 and < image_width.

- **conv_activation_l1:**
	- Represents the type of actiavtion function used in Conv layer 1. 
	- linear, softmax, binary, sigmoid.

- **no_of_filters_l2:**
	- Indicates the number of filters to be included in the Conv layer 2 of the ConvNet model.
	- no_of_filters > 0 and < image_width.

- **conv_activation_l2:**
	- Represents the type of activation function used in Conv layer 2.
	- linear, softmax, binary, sigmoid.

- **no_of_filters_l3:**
	- Indicates the number of filters to be included in the Conv layer 3 of the ConvNet model.
	- no_of_filters > 0 and < -image_width.

- **conv_activation_l3:**
	- Represents the type of activation function used in Conv layer 3. 
	- linear, softmax, binary, sigmoid.

- **no_of_filters_l4:**
	- Indicates the number of filters to be included in the Conv layer 4 of the ConvNet model.
	- no_of_filters > 0 and < image_width.

- **dense_activation_l1:**
	- Represents the type of activation function used in dense layer 1.
	- Example: linear, softmax, binary, sigmoid.

- **dense_activation_l2:**
	- Represents the type of activation function used in dense layer 2. 
	- linear, softmax, binary, sigmoid.

- **relu_alpha:**
	- Specifies the alpha value of relu layer.
	- relu_alpha > 0 and <= 1

- **pool_size_row:**
	- Indicates the row size of the pooling window.
	- pool_size_row > 0 and < input_image_width 

- **pool_size_column:**
	- Specifies the colmn size of the pooling window.
	- pool_size_row > 0 and < input_image_width 

- **dropout:**
	- Specifies whether the dropout layers need to be included or not in the ConvNet model.
	- true or false.

- **dropout_probability_l1:**
	- Represents the dropout probability of dropout layer 1.
	- dropout_probability_l1 ranges from > 0 to < 1.

- **dropout_probability_l2:**
	-Represents the dropout probability of dropout layer 2. 
	- dropout_probability_l1 ranges from > 0 to < 1.

- **dropout_probability_l3:**
	- Specifies the dropout probability of dropout layer 3.
	- dropout_probability_l1 ranges from > 0 to < 1.

- **compile_loss:**
	- Indicates the type of compile_loss used during configuration of the model.
	- categorical_crossentropy.

- **compile_optimizer:**
	- Indicates the type of compile_optimizer used during configuration of the model.
	- adam, adagrad, SGD, adamax *etc.*

- **compile_metrics1:**
	- Indicates the type of compile_metrics used during configuration of the model.
	- accuracy.

- **test_size:**
	- Specifies the partition of data into two parts -> 1) Designed for training 2) Designed for Validation.
	- test_size ranges from > 0 to < 1.

- **num_epochs:**
	- Specifies the number of times the model to be learned during the learning/training process.
	- Optimum range is between 1 to 20.

- **batch_size:**
	- Specifies the number of training samples utilised in one iteration. 
	- batch_size > 0 and < No. Of image the dataset.

- **fit_verbose:**
	- Specifies the option for producing the detailed logging information during the training of a model.
   	- 'verbose=0' will show you nothing (silent).
	- 'verbose=1' will show you an animated progress bar.
	- 'verbose=2' will just Mentions the number of epoch.

- **evaluate_verbose:**
	- Specifies the option for producing the detailed logging information during the evaluation of a model.	
   	- 'verbose=0' will show you nothing (silent).
	- 'verbose=1' will show you an animated progress bar.
	- 'verbose=2' will just Mentions the number of epoch.

- **predict_verbose:**
	- Indicates the option for producing the detailed logging information during the prediction of a model.
   	- 'verbose=0' will show you nothing (silent).
	- 'verbose=1' will show you an animated progress bar.
	- 'verbose=2' will just Mentions the number of epoch.


