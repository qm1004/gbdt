package gbdt

import (
	"fmt"
)

//package main
type Config struct {
	number_of_feature      int
	max_depth              int
	tree_count             int
	shrinkage              float32
	feature_sampling_ratio float32
	data_sampling_ratio    float32
	min_leaf_size          int //min number of sample in leaf
	losstype               int //LOG_LIKEHOOD=1,LEAST_SQUARE=2
	debug                  bool
}

var Conf *Config

func init() {
	Conf = &Config{}
	Conf.number_of_feature = 17
	Conf.max_depth = 5
	Conf.tree_count = 10
	Conf.shrinkage = 0.1
	Conf.feature_sampling_ratio = 1.0
	Conf.data_sampling_ratio = 0.6
	Conf.min_leaf_size = 100
	Conf.losstype = LOG_LIKEHOOD
	Conf.debug = true
}

func (Conf *Config) String() string {
	s := "number_of_feature:" + fmt.Sprintf("%v\n", Conf.number_of_feature)
	s += "max_depth:" + fmt.Sprintf("%v\n", Conf.max_depth)
	s += "tree_count:" + fmt.Sprintf("%v\n", Conf.tree_count)
	s += "shrinkage:" + fmt.Sprintf("%v\n", Conf.shrinkage)
	s += "feature_sampling_ratio:" + fmt.Sprintf("%v\n", Conf.feature_sampling_ratio)
	s += "data_sampling_ratio:" + fmt.Sprintf("%v\n", Conf.data_sampling_ratio)
	s += "min_leaf_size:" + fmt.Sprintf("%v\n", Conf.min_leaf_size)
	s += "losstype:" + fmt.Sprintf("%v\n", Conf.losstype)
	s += "debug:" + fmt.Sprintf("%v\n", Conf.debug)

	return s
}
