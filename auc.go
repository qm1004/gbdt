package gbdt

import (
	"sort"
	"fmt"
)
var _ = fmt.Println

type Auc struct {
	confusion_table []int
	positive_scores []float64
	negative_scores []float64
	threshold       float64
}

func NewAuc() *Auc {
	auc := &Auc{
		confusion_table: make([]int, 4),
		positive_scores: make([]float64, 0),
		negative_scores: make([]float64, 0),
		threshold:       0.5,
	}
	return auc
}

func (self *Auc) Add(score float64, label int) {
	if label > 0 {
		self.positive_scores = append(self.positive_scores, score)
		if score >= self.threshold {
			self.confusion_table[0] += 1 //1->1
		} else {
			self.confusion_table[1] += 1 //1->-1
		}
	} else {
		self.negative_scores = append(self.negative_scores, score)
		if score >= self.threshold {
			self.confusion_table[2] += 1 //-1->1
		} else {
			self.confusion_table[3] += 1 //-1->1
		}
	}
}

func (self *Auc) CalculateAuc() float64{
	sort.Sort(sort.Float64Slice(self.positive_scores))
	sort.Sort(sort.Float64Slice(self.negative_scores))
	n0 := len(self.negative_scores)
	n1 := len(self.positive_scores)
	var rank float64 = 1
	var rankSum float64 = 1
	i0 := 0
	i1 := 0
	for ;i0 < n0 && i1 < n1; {
		v0 := self.negative_scores[i0]
		v1 := self.positive_scores[i1]
		if v0 < v1 {
			i0++
			rank++
		} else if v1 < v0 {
			i1++
			rankSum += rank
			rank++
		} else {
			var tieScore float64 = v0
			k0 := 0
			for ;i0 < n0 && self.negative_scores[i0] == tieScore; {
				k0++
				i0++
			}
			k1 := 0
			for ;i1 < n1 && self.positive_scores[i1] == tieScore; {
				k1++
				i1++
			}
			rankSum += (rank + float64(k0+k1-1)/2.0) * float64(k1)
		}

	}
	if i1 < n1 {
		rankSum += (rank + float64(n1-i1-1)/2.0) * float64(n1-i1)
	}
	return (rankSum/float64(n1) - float64(n1+1)/2) / float64(n0)

}
