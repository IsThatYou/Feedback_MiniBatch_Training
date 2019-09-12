# Feedback_MiniBatch_Training
One of the projects I have done at Comcast Applied AI Lab. This training scheme would guarantee low losses for all patterns for synthetic data
In Feedback mini-batch training, the distribution of patterns within a minibatch is dynamically changed to ensure that all the patterns are optimally trained. If a specific pattern is difficult to learn, it is more likely to be found within a minibatch.​
This method increases the performance of the model. More importantly it guarantees that no patterns are undertrained, which provides a common ground for further improvements/interpretation.
## To Run
You need to install corti-dnlp and corti-data-manager in your virtual enviroment.
## Training
`tf_adversarial_minibatching.py` is the main file to run, remember to specify parameters and directories.
`generate_data.py` can generate data based on given distribution. This is for testing, and traing a normal model.
`test_minibatching.py` test the performance of minibatch trained model on a test dataset.
`view_pattern.py,test.py,test1.py` tests various properties of the model
`plotting.py,plot_dist.py,plotting_scatter.py` plots loss distributions or just distributions
`train_normal.py` trains a normal model for comparision