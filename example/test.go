package main

import (
	"fmt"
	"github.com/qm1004/gbdt"
	"log"
	"runtime"
	"io/ioutil"
	"time"
	"github.com/davecheney/profile"
)

func init() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	runtime.GOMAXPROCS(runtime.NumCPU())
}

func main() {
	defer profile.Start(profile.CPUProfile).Stop()    //monitor program performance
	if gbdt.Conf==nil {
		fmt.Println("nil pointer")
	}else{
		fmt.Println(gbdt.Conf)
	}
	modelname:="./gbdt.model"
	start:=time.Now()
	trainpath:="./train.data"
	train_sample_number:=4584
	dataset:=&gbdt.DataSet{}
	dataset.LoadDataFromFile(trainpath,train_sample_number)
	g:=gbdt.NewGBDT()
	g.Init(dataset)
	g.Train(dataset)
	model:=g.Save()
	if err:=ioutil.WriteFile(modelname,[]byte(model),0666);err!=nil{
		fmt.Println(err)
	}
	latency:=time.Since(start)
	fmt.Println("train time:",latency)

	start2:=time.Now()
	testpath:="./test.data"
	test_sample_number:=4584
	testdataset:=&gbdt.DataSet{}
	testdataset.LoadDataFromFile(testpath,test_sample_number)

	/*model,err:=ioutil.ReadFile(modelname)
	if err!=nil {
		log.Fatal(err)
	}
	g:=&gbdt.GBDT{}
	g.Load(string(model))*///load model from local file 

	g.Load(model)
	samples:=testdataset.GetSamples()
	var T,F float32=0,0
	tree_count:=gbdt.Conf.GetTreecount()
	for i := 0; i < len(samples); i++ {
		pre:=gbdt.LogitCtr(g.Predict(samples[i],tree_count))
		var label int=0
		if pre>=0.5 {
			label=1
		}else{
			label=-1
		}
		if label==samples[i].GetLabel(){
			T+=1
		}else{
			F+=1
		}
	}
	fmt.Println("precision rate:",T/float32(test_sample_number))

	latency=time.Since(start2)
	fmt.Println("precision time:",latency)
}
