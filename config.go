package gbdt

import (
	"fmt"
)

//package main
type Config struct {
	Number_of_feature      int
	Max_depth              int
	Tree_count             int
	Shrinkage              float32
	Feature_sampling_ratio float32
	Data_sampling_ratio    float32
	Min_leaf_size          int //min number of sample in leaf
	Losstype               int //LOG_LIKEHOOD=1,LEAST_SQUARE=2
	Debug                  bool
}

var Conf *Config

func init() {
	Conf = &Config{}
	Conf.Number_of_feature = 17
	Conf.Max_depth = 5
	Conf.Tree_count = 100
	Conf.Shrinkage = 0.1
	Conf.Feature_sampling_ratio = 1
	Conf.Data_sampling_ratio = 0.7
	Conf.Min_leaf_size = 200
	Conf.Losstype = LOG_LIKEHOOD
	Conf.Debug = false
}

func (Conf *Config) GetTreecount() int {
	return Conf.Tree_count
}
func (Conf *Config) String() string {
	s := "number_of_feature:" + fmt.Sprintf("%v\n", Conf.Number_of_feature)
	s += "max_depth:" + fmt.Sprintf("%v\n", Conf.Max_depth)
	s += "tree_count:" + fmt.Sprintf("%v\n", Conf.Tree_count)
	s += "shrinkage:" + fmt.Sprintf("%v\n", Conf.Shrinkage)
	s += "feature_sampling_ratio:" + fmt.Sprintf("%v\n", Conf.Feature_sampling_ratio)
	s += "data_sampling_ratio:" + fmt.Sprintf("%v\n", Conf.Data_sampling_ratio)
	s += "min_leaf_size:" + fmt.Sprintf("%v\n", Conf.Min_leaf_size)
	s += "losstype:" + fmt.Sprintf("%v\n", Conf.Losstype)
	s += "debug:" + fmt.Sprintf("%v\n", Conf.Debug)

	return s
}
