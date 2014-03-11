package gbdt

//package main
type Config struct {
	number_of_feature      int
	max_depth              int
	tree_count             int
	shrinkage              int
	feature_sampling_ratio float32
	data_sampling_ratio    float32
	min_leaf_size          int //min number of sample in leaf
	losstype               int //LOG_LIKEHOOD=1,LEAST_SQUARE=2
	debug                  bool
}

var conf Config

func init() {
	conf.number_of_feature = 100
	conf.max_depth = 5
	conf.tree_count = 100
	conf.losstype = LOG_LIKEHOOD
}
