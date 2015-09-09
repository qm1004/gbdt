package gbdt

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"
	"strings"
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
	Feature_costs          []float32
	Enable_feature_tunning bool
	Debug                  bool
}

var Conf *Config

func init() {
	Conf = &Config{}
	Conf.Number_of_feature = 45
	Conf.Max_depth = 4
	Conf.Tree_count = 100
	Conf.Shrinkage = 0.1
	Conf.Feature_sampling_ratio = 0.7
	Conf.Data_sampling_ratio = 0.6
	Conf.Min_leaf_size = 20000
	Conf.Losstype = LOG_LIKEHOOD
	Conf.Enable_feature_tunning = false
	Conf.Debug = true
}

/*func init() {
	Conf = &Config{}
	Conf.Number_of_feature = 38
	Conf.Max_depth = 5
	Conf.Tree_count = 50
	Conf.Shrinkage = 0.1
	Conf.Feature_sampling_ratio = 1
	Conf.Data_sampling_ratio = 0.7
	Conf.Min_leaf_size = 20000
	Conf.Losstype = LOG_LIKEHOOD
	Conf.Debug = true
}*/

func (Conf *Config) InitFeatureCost() {
    Conf.Feature_costs = make([]float32, Conf.Number_of_feature)
	for i := 0; i < Conf.Number_of_feature; i++ {
		Conf.Feature_costs[i] = 1.0
	}
}
func (Conf *Config) LoadFeatureCost(cost_file string) {
	Conf.Feature_costs = make([]float32, Conf.Number_of_feature)
	for i := 0; i < Conf.Number_of_feature; i++ {
		Conf.Feature_costs[i] = 1.0
	}
	f, err := os.Open(cost_file)
	defer f.Close()
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	br := bufio.NewReader(f)
	for {
		line, err := br.ReadString('\n')
		if err == io.EOF {
			log.Println("cost_file load done!")
			break
		}
		line = strings.TrimSpace(line)
		if strings.HasPrefix(line, "#") {
			continue
		}
		kv := strings.Split(line, ":")
		fid, err := strconv.Atoi(kv[0])
		if err != nil {
			// handle error
			log.Println("feature paser err", err, kv)
			os.Exit(2)
		}
		val, err := strconv.ParseFloat(kv[1], 32)
		if err != nil {
			// handle error
			log.Println("feature paser err", err, kv)
			os.Exit(2)
		}
		if fid < Conf.Number_of_feature {
			Conf.Feature_costs[fid] = float32(val)
		}

	}
	Conf.Enable_feature_tunning = true
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
	s += "Enable_feature_tunning:" + fmt.Sprintf("%v\n", Conf.Enable_feature_tunning)
	s += "debug:" + fmt.Sprintf("%v\n", Conf.Debug)

	return s
}
