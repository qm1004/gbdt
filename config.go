package gbdt

import (
	"fmt"
)
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
	conf.number_of_feature = 17
	conf.max_depth = 5
	conf.tree_count = 10
	conf.shrinkage = 0.1
	conf.feature_sampling_ratio =1.0
	conf.data_sampling_ratio = 0.6
	conf.min_leaf_size = 100
	conf.losstype = LOG_LIKEHOOD
	conf.debug = 1
}


func (conf *Config) String()  string{
	s:="number_of_feature:"+fmt.Sprintf("%v\n",conf.number_of_feature)
	s+="max_depth:"+fmt.Sprintf("%v\n",conf.max_depth)
	s+="tree_count:"+fmt.Sprintf("%v\n",conf.tree_count)
	s+="shrinkage:"+fmt.Sprintf("%v\n",conf.shrinkage)
	s+="feature_sampling_ratio:"+fmt.Sprintf("%v\n",conf.feature_sampling_ratio)
	s+="data_sampling_ratio:"+fmt.Sprintf("%v\n",conf.data_sampling_ratio)
	s+="min_leaf_size:"+fmt.Sprintf("%v\n",conf.min_leaf_size)
	s+="losstype:"+fmt.Sprintf("%v\n",conf.losstype)
	s+="debug:"+fmt.Sprintf("%v\n",conf.debug)

	return s
}