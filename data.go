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

const (
	ITEMSPLIT         = " "
	FEATURESCORESPLIT = ":"
)

type Feature struct {
	Id    int
	Value float32
	//is_continueous bool //1:CONTINUOUS_FEATURE,0:DISCRETE_FEATURE
}

type Sample struct {
	Features []float32
	label    int
	target   float32
	weight   float32
	Treenum  int
	pred     float32
}

/*func (self *Sample) FindFeature(id int) (int, bool) {
	l := 0
	h := len(self.Features) - 1
	for l <= h {
		index := (l + h) / 2
		if id == self.Features[index].Id {
			return index, true
		} else if id > self.Features[index].Id {
			l = index + 1
		} else {
			h = index - 1
		}
	}
	return -1, false
}*/

func (self *Sample) GetLabel() int {
	return self.label
}
func (self *Sample) GetWeight() float32 {
	return self.weight
}

type MapSample struct {
	label   int
	target  float32
	weight  float32
	feature map[int]float32
}

type DataSet struct {
	Samples []*Sample
	//max_number int //feature dimensions
}

func (d *DataSet) GetSamples() []*Sample {
	return d.Samples
}

func (d *DataSet) FromString(line string, row int) {
	//fmt.Println("line:",line)
	items := strings.Split(line, ITEMSPLIT)
	//fmt.Println("items:",items,items[0],items[1],len(items))
	d.Samples[row] = &Sample{}
	d.Samples[row].Treenum = -1
	if weight, err := strconv.ParseFloat(items[0], 32); err != nil {
		log.Println("weight paser err:", items[0], err, row)
		os.Exit(1)
	} else {
		//fmt.Println("Samples:",len(d.Samples))
		d.Samples[row].weight = float32(weight)
	}
	if label, err := strconv.Atoi(items[1]); err != nil {
		log.Println("label paser err:", items[1], err, row)
		os.Exit(1)
	} else {
		d.Samples[row].label = label
	}
	d.Samples[row].Features = make([]float32, Conf.Number_of_feature)
	for i := 0; i < Conf.Number_of_feature; i++ {
		d.Samples[row].Features[i] = UNKNOWN_VALUE
	}
	for i := 2; i < len(items); i++ {
		kv := strings.Split(items[i], FEATURESCORESPLIT)
		fid, err := strconv.Atoi(kv[0])
		if err != nil {
			// handle error
			log.Println("feature paser err", items[i], err, row, kv)
			os.Exit(2)
		}
		val, err := strconv.ParseFloat(kv[1], 32)
		if err != nil {
			// handle error
			log.Println("feature paser err", items[i], err, row, kv)
			os.Exit(2)
		}
		if fid < Conf.Number_of_feature {
			d.Samples[row].Features[fid] = float32(val)
		}
	}
}

func (d *DataSet) LoadDataFromFileWeight(path string, sample_number int, ignoreweight bool) {
	f, err := os.Open(path)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	d.Samples = make([]*Sample, sample_number)
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
			d.Samples[row].weight = 1
		}
		row++
		if row%200000 == 1 {
			fmt.Println("read ", row, " rows")
		}
	}
	//fmt.Println("load data done!", len(d.Samples), d.Samples[0])

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
