package gbdt

import (
	"math"
	"strings"
	"strconv"
)

type GBDT struct {
	trees      []*RegressionTree
	tree_count int
	shrinkage  float32
	bias       float32
}

func NewGBDT() (gbdt *GBDT) {
	gbdt = &GBDT{
		trees:      nil,
		tree_count: conf.tree_count,
		shrinkage:  conf.shrinkage,
		bias:       0,
	}
	for i := 0; i < conf.tree_count; i++ {
		tree := NewRegressionTree()
		gbdt.trees = append(gbdt.trees, tree)
	}
}

func (self *GBDT) Init(d *DataSet) {
	var s float32 = 0
	var c float32 = 0
	for _, sample := range d.samples {
		s += float32(sample.label) * sample.weight
		c += sample.weight
	}
	y_avg := s / c
	if conf.losstype == LEAST_SQUARE {
		self.bias = y_avg
	} else if conf.losstype == LOG_LIKIHOOD {
		self.bias = float32(math.Log(float64((1+y_avg)/(1-y_avg))) / 2.0)
	}
	self.trees = make([]*RegressionTree, conf.tree_count)
}

func (self *GBDT) Train(d *DataSet) {
	var sample_number int = len(DataSet.samples)
	if conf.data_sampling_ratio < 1 {
		sample_number = int(conf.data_sampling_ratio * float32(len(DataSet.samples)))
	}
	self.Init(d, conf)

	for i := 0; i < conf.tree_count; i++ {
		if conf.data_sampling_ratio < 1 {
			random_shuffle(d.samples, len(d.samples))
		}

		for j := 0; j < sample_number; j++ {
			p := self.Predict(d.samples[j], i)
			d.sampels[j].target = FxGradient(d.samples[j].label, p)
		}
		if conf.debug {
			//cal auc
			//cal loss
		}
		self.trees[i].Fit(d, sample_number)
	}

}

func (self *GBDT) Predict(sample *Sample, n int) float32 {
	if trees == nil {
		return UNKNOWN_VALUE
	}
	r := self.bias
	for i := 0; i < n; i++ {
		r += self.shrinkage * self.trees[i].Predict(sample)
	}
	return r
}

func (self *GBDT) Save() string{
	vs:=make([]string,0)
	vs=append(vs,strings.FormatFloat(self.shrinkage,"f",32))
	vs=append(vs,strings.FormatFloat(self.bias,"f",32))
	for i := 0; i < self.tree_count; i++ {
		vs=append(vs,self.trees[i].Save())
	}
	return strings.Join(vs,"\n;\n")
}

func (self *GBDT) Load(s string) {
	self.trees=nil
	var err error
	vs:=strings.Split(s,"\n;\n")
	self.tree_count = len(vs)-2
	if self.shrinkage,err = strconv.ParseFloat(vs[0],32);err{
		log.Fatal("shrinkage",err)
	}
	if self.bias,err = strconv.ParseFloat(vs[1],32);err {
		log.Fatal("bias",err)
	}
	self.trees=make([]*RegressionTree,self.tree_count)

	for i := 0; i < self.tree_count; i++ {
		self.trees[i].Load(vs[i+2])
	}

}