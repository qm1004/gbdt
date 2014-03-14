package main

import (
	"fmt"
	"github.com/qm1004/gbdt"
	"log"
	"runtime"
	"io/ioutil"
)

func init() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	runtime.GOMAXPROCS(runtime.NumCPU())
}

func main() {
	if gbdt.Conf==nil {
		fmt.Println("nil pointer")
	}else{
		fmt.Println(gbdt.Conf)
	}
	path:="/Users/minqian/Desktop/tmp/train.data"
	//modelname:="opt/tmp/gbdt/gbdt.model"
	modelname:="/Users/minqian/Desktop/tmp/gbdt.model"
	sample_number:=2000
	dataset:=&gbdt.DataSet{}
	dataset.LoadDataFromFile(path,sample_number)

	g:=gbdt.NewGBDT()
	g.Init(dataset)
	g.Train(dataset)
	model:=g.Save()
	if err:=ioutil.WriteFile(modelname,[]byte(model),0777);err!=nil{
		fmt.Println(err)
	}
}
