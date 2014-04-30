package gbdt

import (
	"fmt"
	"log"
	"math"
	"os"
	"strconv"
	"strings"
	"time"
)

var _ = time.Now
var _ = os.Exit

type GBDT struct {
	trees      []*RegressionTree
	tree_count int
	shrinkage  float32
	bias       float32
}

func NewGBDT() (gbdt *GBDT) {
	gbdt = &GBDT{
		trees:      make([]*RegressionTree, 0),
		tree_count: Conf.Tree_count,
		shrinkage:  Conf.Shrinkage,
		bias:       0,
	}
	for i := 0; i < Conf.Tree_count; i++ {
		tree := NewRegressionTree()
		gbdt.trees = append(gbdt.trees, tree)
	}
	//fmt.Println("gbdt:",len(gbdt.trees))
	return
}

func (self *GBDT) Init(d *DataSet) {
	var s float32 = 0
	var c float32 = 0

	i := 0
	for _, sample := range d.samples {
		s += float32(sample.label) * (sample.weight)
		c += sample.weight
		i++
	}
	y_avg := s / c
	if Conf.Losstype == LEAST_SQUARE {
		self.bias = y_avg
	} else if Conf.Losstype == LOG_LIKEHOOD {
		self.bias = float32(math.Log(float64((1+y_avg)/(1-y_avg))) / 2.0)
	}
	//self.trees = make([]*RegressionTree, Conf.Tree_count)
}

func (self *GBDT) Train(d *DataSet) {

	var sample_number int = len(d.samples)
	if Conf.Data_sampling_ratio < 1 {
		sample_number = int(Conf.Data_sampling_ratio * float32(len(d.samples)))
	}
	self.Init(d)

	for i := 0; i < Conf.Tree_count; i++ {
		fmt.Printf("iteration:%d ", i)
		start := time.Now()

		if Conf.Data_sampling_ratio < 1 {
			if !random_shuffle(d.samples, len(d.samples)) {
				fmt.Println("not random_shuffle!")
				os.Exit(1)
			}
		}
		for j := 0; j < sample_number; j++ {
			p := self.Predict(d.samples[j], i)
			d.samples[j].target = FxGradient(d.samples[j].label, p)
		}
		if Conf.Debug {
			//cal auc
			auc := NewAuc()
			for j := 0; j < len(d.samples); j++ {
				p := LogitCtr(self.Predict(d.samples[j], i))
				auc.Add(float64(p), float64(d.samples[j].weight), d.samples[j].label)
			}
			fmt.Println("auc:", auc.CalculateAuc())
			//cal loss
			/*var s, c float64 = 0, 0
			for j := 0; j < len(d.samples); j++ {
				p := self.Predict(d.samples[j], i)
				s += float64(Float32Square(float32(d.samples[j].label)-p) * d.samples[j].weight)
				c += float64(d.samples[j].weight)
			}
			fmt.Println("rmse:", math.Sqrt(s/c))*/

		}
		self.trees[i].Fit(d, sample_number)
		latency := time.Since(start)
		fmt.Println("latency:", latency)
	}

}

func (self *GBDT) Predict(sample *Sample, n int) float32 {
	if self.trees == nil {
		return UNKNOWN_VALUE
	}
	if n == 0 {
		sample.treenum++
		sample.pred = self.bias
		return sample.pred
	}
	if sample.treenum == -1 {
		sample.pred = self.bias
		sample.treenum++
	}
	for i := sample.treenum; i < n; i++ {
		sample.pred += self.shrinkage * self.trees[i].Predict(sample)
	}
	sample.treenum = n
	return sample.pred
}

func (self *GBDT) Save() string {
	vs := make([]string, 0)
	vs = append(vs, strconv.FormatFloat(float64(self.shrinkage), 'f', 4, 32))
	vs = append(vs, strconv.FormatFloat(float64(self.bias), 'f', 4, 32))
	for i := 0; i < self.tree_count; i++ {
		vs = append(vs, self.trees[i].Save())
	}
	return strings.Join(vs, "\n;\n")
}

func (self *GBDT) Load(s string) {
	self.trees = nil
	vs := strings.Split(s, "\n;\n")
	self.tree_count = len(vs) - 2
	if tempshrinkage, err := strconv.ParseFloat(vs[0], 32); err != nil {
		log.Fatal("shrinkage", err)
	} else {
		self.shrinkage = float32(tempshrinkage)
	}

	if tempbias, err := strconv.ParseFloat(vs[1], 32); err != nil {
		log.Fatal("bias", err)
	} else {
		self.bias = float32(tempbias)
	}

	self.trees = make([]*RegressionTree, self.tree_count)

	for i := 0; i < self.tree_count; i++ {
		self.trees[i] = NewRegressionTree()
		self.trees[i].Load(vs[i+2])
	}

}

func (self *GBDT) GetFeatureWeight() PairList {
	feature_weight := make(map[int]float32)
	for i := 0; i < self.tree_count; i++ {
		tree_feature_weight := self.trees[i].GetTreeFeatureWeight()
		for key, val := range tree_feature_weight {
			feature_weight[key] += val
		}
	}
	feature_weight_list := SortMapByValue(feature_weight)
	return feature_weight_list
}
