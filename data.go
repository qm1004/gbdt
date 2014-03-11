package gbdt

type Feature struct {
	id    int
	value float32
	//is_continueous bool //1:CONTINUOUS_FEATURE,0:DISCRETE_FEATURE
}

type Sample struct {
	feature []Feature
	label   int
	target  float32
	weight  float32
}

func (self *Sample) ToMapSample() *MapSample {
	m := &MapSample{
		label:   self.label,
		target:  self.target,
		weight:  self.weight,
		feature: make(map[int]float32),
	}
	for _, v := range self.feature {
		m.feature[v.id] = v.value
	}
	return m
}

type DataSet struct {
	samples []*Sample
	//max_number int //feature dimensions
}

type MapSample struct {
	label    int
	target   float32
	weight   float32
	fealture map[int]float32
}

type Tuple struct {
	value  float32
	target float32
	weight float32
}

type TupleList struct {
	tulpelist []Tuple
}

func (self *TupleList) Len() int {
	return len(self.tulpelist)
}

func (self *TupleList) Swap(i, j int) {
	self.tulpelist[i], self.tulpelist[j] = self.tulpelist[j], self.tulpelist[i]
}

func (self *TupleList) Less(i, j int) bool {
	return self.tulpelist[i].value <= self.tulpelist[j].value
}

func (tp *TupleList) AddTuple(value, target, weight float32) {
	temp := Tuple{
		value:  value,
		target: target,
		weight: weight,
	}
	tp.tulpelist = append(tp.tulpelist, temp)
}
func NewTupleList() *TupleList {
	tp := &TupleList{}
	tp.tulpelist = []Tuple{}
	return tp
}
