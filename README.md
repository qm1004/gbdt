gbdt
====

GBDT:Gradient Boosting Decision Tree implemented by golang.<br/>

#Characteristic
Concurrency:Implementing local cocurrent with goroutine when building single tree at each iteration.<br/>
Support missing feature value.Every node in the tree has a child named "UNKNOWN VALUE".So every tree is the ternary tree<br/> 

#Data format
1 1 0:1.1446 1:2 2:35 3:0 4:206.0 ....<br/>
weight label featureid:feature_value....<br/>

#Parameter
All gbdt parameter define in config.go.

Number_of_feature:<br/>
Feature dimensions.eg:In example/train.data,there are 17 different features.

Max_depth:<br/>
The maximal depth of the tree. 

Tree_count:<br/>
Tree count ,namely  iteration numbers.

Shrinkage:<br/>
Step size.

Feature_sampling_ratio:<br/>
Feature sample ratio for build tree.

Data_sampling_ratio:<br/>
Data sample ration for build tree.

Min_leaf_size:<br/>
Minimal number of samples in leaf.

Losstype :<br/>
Loss function type,eg least square,log likehood

Debug :<br/>
Print some debug info.

Enable_feature_tunning:<br/>
[default=false].<br/>
If Enable_feature_tunning is true, it can reduce frenquency of feature which is always chosen as split<br/> feature.Let more feature appear in the tree models.<br/>
