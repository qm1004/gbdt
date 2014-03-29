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
	//trainpath:="/opt/tmp/gbdt/data/train.data"
	train_sample_number := 4584
	//train_sample_number:=4186052
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
	//testpath:="/opt/tmp/gbdt/data/smalltest.data"
	//test_sample_number := 837210
	test_sample_number:=4584
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
	tree_count := gbdt.Conf.GetTreecount()
	var click_sum,no_click_sum float64
	for i := 0; i < len(samples); i++ {
		if samples[i].GetLabel()==1 {
			click_sum+=float64(samples[i].GetWeight())
		}else{
			no_click_sum+=float64(samples[i].GetWeight())
		}
	}
	latency = time.Since(start2)
	fmt.Println("precision time:", latency)
	auc := gbdt.NewAuc()
	for j := 0; j < len(samples); j++ {
		p := gbdt.LogitCtr(g.Predict(samples[j], tree_count))
		auc.Add(float64(p), float64(samples[j].GetWeight()),samples[j].GetLabel())
	}
	fmt.Println("auc:", auc.CalculateAuc())
	auc.PrintConfusionTable()
	/*FeatureMapFile:="./feature.map"
	feature_data,err:=ioutil.ReadFile(FeatureMapFile)
	if err!=nil {
		log.Fatal(err)
	}
	feature_map:=gbdt.LoadFeatureMap(string(feature_data))
	feature_weight_list:=g.GetFeatureWeight()
	for i := 0; i < len(feature_weight_list); i++ {
		fid:=feature_weight_list[i].Key
		fmt.Println(feature_map[fid],":",feature_weight_list[i].Value)
	}*/
		
}
