package gbdt

import (
	"fmt"
	"sort"
)

var _ = fmt.Println

type weight_score struct {
	score  float64
	weight float64
}

type WeightScoreList []*weight_score

func (p WeightScoreList) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }
func (p WeightScoreList) Len() int           { return len(p) }
func (p WeightScoreList) Less(i, j int) bool { return p[i].score < p[j].score }

type Auc struct {
	confusion_table []int
	positive_scores WeightScoreList
	negative_scores WeightScoreList
	threshold       float64
}

func NewAuc() *Auc {
	auc := &Auc{
		confusion_table: make([]int, 4),
		positive_scores: make(WeightScoreList, 0),
		negative_scores: make(WeightScoreList, 0),
		threshold:       0.5,
	}
	return auc
}

func (self *Auc) Add(score float64, weight float64, label int) {
	if label > 0 {
		ws := &weight_score{score: score, weight: weight}
		self.positive_scores = append(self.positive_scores, ws)
		if score >= self.threshold {
			self.confusion_table[0] += 1 //1->1
		} else {
			self.confusion_table[1] += 1 //1->-1
		}
	} else {
		ws := &weight_score{score: score, weight: weight}
		self.negative_scores = append(self.negative_scores, ws)
		if score >= self.threshold {
			self.confusion_table[2] += 1 //-1->1
		} else {
			self.confusion_table[3] += 1 //-1->1
		}
	}
}
func (self *Auc) PrintConfusionTable() {
	fmt.Println(self.confusion_table[0], "\t", self.confusion_table[1])
	fmt.Println(self.confusion_table[2], "\t", self.confusion_table[3])
}

//reference
//http://blog.csdn.net/cserchen/article/details/7535182
func (self *Auc) CalculateAuc() float64 {
	sort.Sort(sort.Reverse(self.positive_scores))
	sort.Sort(sort.Reverse(self.negative_scores))
	//fmt.Println("score:",self.positive_scores[0].score,self.positive_scores[1].score,self.positive_scores[20].score)
	//fmt.Println("neg score:",self.negative_scores[0].score,self.negative_scores[1].score,self.negative_scores[20].score)
	i0 := 0
	i1 := 0
	n0 := len(self.negative_scores)
	n1 := len(self.positive_scores)
	var auc_temp float64 = 0.0
	var click_sum float64 = 0.0
	var old_click_sum float64 = 0.0
	var no_click float64 = 0.0
	var no_click_sum float64 = 0.0
	var lastscore float64 = 2.0
	for i0 < n0 && i1 < n1 {
		v0 := self.negative_scores[i0].score
		w0 := self.negative_scores[i0].weight
		v1 := self.positive_scores[i1].score
		w1 := self.positive_scores[i1].weight
		var label int = 0
		var v float64
		if v1 > v0 {
			i1++
			label = 1
			v = v1
		} else if v1 < v0 {
			i0++
			label = -1
			v = v0
		} else {
			v = v1
			i0++
			for i0 < n0 && self.negative_scores[i0].score == v {
				w0 += self.negative_scores[i0].weight
				i0++
			}
			i1++
			for i1 < n1 && self.positive_scores[i1].score == v {
				w1 += self.positive_scores[i1].weight
				i1++
			}
		}
		if lastscore != v {
			auc_temp += (click_sum + old_click_sum) * no_click / 2.0
			old_click_sum = click_sum
			no_click = 0.0
			lastscore = v
		}
		if label == 1 {
			click_sum += w1
		} else if label == -1 {
			no_click += w0
			no_click_sum += w0
		} else {
			no_click += w0
			no_click_sum += w0
			click_sum += w1
		}
	}
	for i1 < n1 {
		v1 := self.positive_scores[i1].score
		w1 := self.positive_scores[i1].weight
		if lastscore != v1 {
			auc_temp += (click_sum + old_click_sum) * no_click / 2.0
			old_click_sum = click_sum
			no_click = 0.0
			lastscore = v1
		}
		click_sum += w1
		i1++
	}
	for i0 < n0 {
		v0 := self.negative_scores[i0].score
		w0 := self.negative_scores[i0].weight
		if lastscore != v0 {
			auc_temp += (click_sum + old_click_sum) * no_click / 2.0
			old_click_sum = click_sum
			no_click = 0.0
			lastscore = v0
		}
		no_click += w0
		no_click_sum += w0
		i0++
	}
	auc_temp += (click_sum + old_click_sum) * no_click / 2.0
	return auc_temp / (click_sum * no_click_sum)
}
