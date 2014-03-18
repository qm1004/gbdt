gbdt
====

GBDT:Gradient Boosting Decision Tree implemented by golang

#Parameter
All gbdt parameter define in config.go.

Number_of_feature:Feature dimensions.eg:In example/train.data,there are 17 different features.

Max_depth:The maximal depth of the tree. 

Tree_count:Tree count ,namely  iteration numbers.

Shrinkage:Step size.

Feature_sampling_ratio:Feature sample ratio for build tree.

Data_sampling_ratio:Data sample ration for build tree.

Min_leaf_size:Minimal number of samples in leaf.

Losstype :Loss function type,eg least square,log likehood

Debug :Print some debug info.
