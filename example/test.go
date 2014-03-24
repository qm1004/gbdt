package main

import (
	"fmt"
	//"github.com/davecheney/profile"
	"github.com/qm1004/gbdt"
	"io/ioutil"
	"log"
	"runtime"
	"time"
)

func init() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	runtime.GOMAXPROCS(runtime.NumCPU())
}

func main() {
	//defer profile.Start(profile.CPUProfile).Stop() //monitor program performance
	if gbdt.Conf == nil {
		fmt.Println("nil pointer")
	} else {
		fmt.Println(gbdt.Conf)
	}
	modelname := "./gbdt.model"
	start := time.Now()
	trainpath := "./train.data"
	//trainpath:="/opt/tmp/gbdt/alltrain.data"
	train_sample_number := 4584
	//train_sample_number:=458334
	dataset := &gbdt.DataSet{}
	dataset.LoadDataFromFile(trainpath, train_sample_number)
	g := gbdt.NewGBDT()
	g.Init(dataset)
	g.Train(dataset)
	model := g.Save()
	if err := ioutil.WriteFile(modelname, []byte(model), 0666); err != nil {
		fmt.Println(err)
	}
	latency := time.Since(start)
	fmt.Println("train time:", latency)

	start2 := time.Now()
	testpath := "./test.data"
	//testpath:="/opt/tmp/gbdt/alltest.data"
	test_sample_number := 4584
	//test_sample_number:=458334
	testdataset := &gbdt.DataSet{}
	testdataset.LoadDataFromFile(testpath, test_sample_number)

	/*model,err:=ioutil.ReadFile(modelname)
	if err!=nil {
		log.Fatal(err)
	}
	g:=&gbdt.GBDT{}
	g.Load(string(model))*/ //load model from local file

	g.Load(model)
	samples := testdataset.GetSamples()
	var T, F float32 = 0, 0
	tree_count := gbdt.Conf.GetTreecount()
	for i := 0; i < len(samples); i++ {
		pre := gbdt.LogitCtr(g.Predict(samples[i], tree_count))
		var label int = 0
		if pre >= 0.5 {
			label = 1
		} else {
			label = -1
		}
		if label == samples[i].GetLabel() {
			T += 1
		} else {
			F += 1
		}
	}
	fmt.Println("precision rate:", T/float32(test_sample_number))

	latency = time.Since(start2)
	fmt.Println("precision time:", latency)
	feature_weight_list:=g.GetFeatureWeight()
	for i := 0; i < len(feature_weight_list); i++ {
		fmt.Println(feature_weight_list[i].Key,":",feature_weight_list[i].Value)
	}
}
