package gbdt

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	//"log"
)

const (
	ITEMSPLIT         = " "
	FEATURESCORESPLIT = ":"
)

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

type MapSample struct {
	label   int
	target  float32
	weight  float32
	feature map[int]float32
}

type DataSet struct {
	samples []*Sample
	//max_number int //feature dimensions
}

func (d *DataSet) FromString(line string, row int) {
	//fmt.Println("line:",line)
	items := strings.Split(line, ITEMSPLIT)
	//fmt.Println("items:",items,items[0],items[1],len(items))
	d.samples[row] = &Sample{}
	if weight, err := strconv.ParseFloat(items[0], 32); err != nil {
		fmt.Println("weight paser err:", items[0], err, row)
		os.Exit(1)
	} else {
		//fmt.Println("samples:",len(d.samples))
		d.samples[row].weight = float32(weight)
	}
	if label, err := strconv.Atoi(items[1]); err != nil {
		fmt.Println("label paser err:", items[1], err, row)
		os.Exit(1)
	} else {
		d.samples[row].label = label
	}

	for i := 2; i < len(items); i++ {
		kv := strings.Split(items[i], FEATURESCORESPLIT)
		fid, err := strconv.Atoi(kv[0])
		if err != nil {
			// handle error
			fmt.Println("feature paser err", items[i], err, row, kv)
			os.Exit(2)
		}
		val, err := strconv.ParseFloat(kv[1], 32)
		if err != nil {
			// handle error
			fmt.Println("feature paser err", items[i], err, row, kv)
			os.Exit(2)
		}
		d.samples[row].feature = append(d.samples[row].feature, Feature{id: fid, value: float32(val)})
	}
}

func (d *DataSet) LoadDataFromFileWeight(path string, sample_number int, ignoreweight bool) {
	f, err := os.Open(path)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	d.samples = make([]*Sample, sample_number)
	defer f.Close()
	br := bufio.NewReader(f)
	row := 0
	for {
		line, err := br.ReadString('\n')
		if err == io.EOF {
			fmt.Printf("read %d rows\n", row)
			break
		}
		line = strings.TrimSpace(line)
		d.FromString(line, row)
		if ignoreweight {
			d.samples[row].weight = 1
		}
		row++
	}
	fmt.Println("load data done!", len(d.samples), d.samples[0])

}

func (d *DataSet) LoadDataFromFile(path string, sample_number int) {
	d.LoadDataFromFileWeight(path, sample_number, false)
}

type Tuple struct {
	value  float32
	target float32
	weight float32
}

type TupleList struct {
	tuplelist []Tuple
}

func (self *TupleList) Len() int {
	return len(self.tuplelist)
}

func (self *TupleList) Swap(i, j int) {
	self.tuplelist[i], self.tuplelist[j] = self.tuplelist[j], self.tuplelist[i]
}

func (self *TupleList) Less(i, j int) bool {
	return self.tuplelist[i].value <= self.tuplelist[j].value
}

func (tp *TupleList) AddTuple(value, target, weight float32) {
	temp := Tuple{
		value:  value,
		target: target,
		weight: weight,
	}
	tp.tuplelist = append(tp.tuplelist, temp)
}
func NewTupleList() *TupleList {
	tp := &TupleList{}
	tp.tuplelist = []Tuple{}
	return tp
}
