package gbdt

import (
	"bufio"
	"io"
	"os"
	"fmt"
	"strings"
	"strconv"
	"log"
)

const (
	ITEMSPLIT = " "
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

type DataSet struct {
	samples []*Sample
	//max_number int //feature dimensions
}
func (d *DataSet) FromString(line string,row int) {
	itmes:=strings.Split(line,ITEMSPLIT)
	if weight,err:=strconv.ParseFloat(items[0],32);err!=nil{
		fmt.Println("weight paser err:",items[0],err,row)
		os.Exit(1)
	}else{
		d.samples[row].weight=float32(weight)
	}

	if label,err:=strconv.Atoi(items[1]);err!=nil {
		fmt.Println("label paser err:",items[1],err,row)
		os.Exit(1)
	}else{
		d.samples[row].label=label
	}

	for i := 2; i < len(items); i++ {
		kv := strings.Split(item[i], ":")
		fid, err := strconv.Atoi(kv[0])
		if err != nil {
			// handle error
			fmt.Println("feature paser err",item[i],err,row)
			os.Exit(2)
		}
		val, err := strconv.ParseFloat(kv[1], 32)
		if err != nil {
			// handle error
			fmt.Println("feature paser err",item[i],err,row)
			os.Exit(2)
		}
		d.samples[row].feature.id=fid
		d.samples[row].feature.value=float32(val)
	}
}

func (d *DataSet) LoadDataFromFile(path string,sample_number int, ignoreweight bool) {
	f,err:=os.Open(path)
	if err!=nil {
		fmt.Println(err)
		os.Exit(1)
	}
	d.samples:=make([]*Sample,sample_number)
	defer f.Close()
	br:=bufio.NewReader(f)
	row:=0
	for {
		line,err:=br.ReadString('\n')
		if err==io.EOF{
			fmt.Printf("read %d rows\n",row)
			break
		}
		d.FromString(line,row)
		if ignoreweight {
			d.samples[row].weight=1
		}
		row++
	}

}

func (d *DataSet) LoadDataFromFile(path string,sample_number int){
	d.LoadDataFromFile(path,sample_number,false)
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


