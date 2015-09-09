package main

import (
	"flag"
	"fmt"
	"github.com/qm1004/gbdt"
	"io/ioutil"
	"log"
	"math"
	"os"
	"runtime"
	"time"
)

func init() {
	log.SetFlags(log.LstdFlags | log.Lshortfile)
	runtime.GOMAXPROCS(runtime.NumCPU())
}

func main() {
	start := time.Now()
	var debug bool
	flag.BoolVar(&debug, "debug", false, "whether print training info")

	var train_sample_number int
	flag.IntVar(&train_sample_number, "trainrows", 0, "train sample number")

	var trainpath, modelname, aucfile string
	flag.StringVar(&trainpath, "trainpath", "/opt/tmp/search_rerank/train.data", "train data path")
	flag.StringVar(&modelname, "modelname", "/opt/tmp/search_rerank/gbdt.model", "model file")
	flag.StringVar(&aucfile, "aucfile", "/opt/tmp/search_rerank/train_auc", "training auc")

	var feature_cost_file, feature_map_file string
	flag.StringVar(&feature_cost_file, "feature_cost_file", "./data/feature.cost", "feature init cost")
	flag.StringVar(&feature_map_file, "feature_map_file", "./data/feature.map", "feature map")

	var treecount int
	flag.IntVar(&treecount, "treecount", 100, "tree number")

	var feature_num int
	flag.IntVar(&feature_num, "feature_num", 45, "feature number")

	var depth, min_leaf_size int
	flag.IntVar(&depth, "depth", 4, "tree depth")
	flag.IntVar(&min_leaf_size, "min_leaf_size", 20000, "min leaf size")

	var feature_sampling_ratio, data_sampling_ratio, shrinkage float64
	flag.Float64Var(&feature_sampling_ratio, "feature_sampling_ratio", 0.7, "feature sampling ratio")
	flag.Float64Var(&data_sampling_ratio, "data_sampling_ratio", 0.6, "data sampling ration")
	flag.Float64Var(&shrinkage, "shrinkage", 0.1, "step size")

	var test_sample_number int
	flag.IntVar(&test_sample_number, "testrows", 0, "test sample number")

	var testpath string
	flag.StringVar(&testpath, "testpath", "/opt/tmp/search_rerank/test.data", "test data path")

	var istestset bool
	flag.BoolVar(&istestset, "istestset", false, "whether use testset")

	var switch_feature_tune bool
	flag.BoolVar(&switch_feature_tune, "switch_feature_tune", false, "switch feature tune")

	flag.Parse()
	/*if train_sample_number<1000000 {
		log.Println("read train file err")
		os.Exit(1)
	}*/

	gbdt.Conf.Debug = debug
	gbdt.Conf.Tree_count = treecount
	gbdt.Conf.Number_of_feature = feature_num
	gbdt.Conf.Max_depth = depth
	gbdt.Conf.Min_leaf_size = min_leaf_size
	gbdt.Conf.Feature_sampling_ratio = float32(feature_sampling_ratio)
	gbdt.Conf.Data_sampling_ratio = float32(data_sampling_ratio)
	gbdt.Conf.Shrinkage = float32(shrinkage)
	gbdt.Conf.Enable_feature_tunning = switch_feature_tune

	if gbdt.Conf.Enable_feature_tunning {
		gbdt.Conf.InitFeatureCost()
	}
	//gbdt.Conf.LoadFeatureCost(feature_cost_file)
	//log.Println(feature_cost_file,":feature cost file load done!")
	if gbdt.Conf == nil {
		log.Println("nil pointer")
		os.Exit(1)
	} else {
		log.Println(gbdt.Conf)
	}

	dataset := &gbdt.DataSet{}
	dataset.LoadDataFromFile(trainpath, train_sample_number)
	g := gbdt.NewGBDT()
	g.Init(dataset)
	g.Train(dataset)
	model := g.Save()
	if err := ioutil.WriteFile(modelname, []byte(model), 0666); err != nil {
		log.Println(err)
		os.Exit(1)
	}
	var auc_score, logloss float32 = 0.0, 0.0
	if istestset {
		testdataset := &gbdt.DataSet{}
		testdataset.LoadDataFromFile(testpath, test_sample_number)
		auc_score, logloss = EvalModel(testdataset, g, treecount)
	} else {
		auc_score, logloss = EvalModel(dataset, g, treecount)
	}

	log.Println("auc:", auc_score)
	log.Println("logloss:", logloss)
	evalscore := fmt.Sprintf("auc:%v,logloss:%v", auc_score, logloss)
	if err := ioutil.WriteFile(aucfile, []byte(evalscore), 0666); err != nil {
		log.Println(err)
		os.Exit(1)
	}
	feature_data, err := ioutil.ReadFile(feature_map_file)
	if err != nil {
		log.Fatal(err)
	}

	feature_map := gbdt.LoadFeatureMap(string(feature_data))
	log.Println(feature_map_file, ":feature_map load done!")

	feature_weight_list := g.GetFeatureWeight()
	log.Println("feature weight:")
	for i := 0; i < len(feature_weight_list); i++ {
		fid := feature_weight_list[i].Key
		log.Println(feature_map[fid], ":", feature_weight_list[i].Value)
	}
    if switch_feature_tune {
        log.Println("feature cost:")
        for i := 0; i < len(gbdt.Conf.Feature_costs); i++ {
            log.Println(i, " ", feature_map[i], ":", gbdt.Conf.Feature_costs[i])
        }
    }

	log.Println("training time:", time.Since(start))

}

func EvalModel(testdataset *gbdt.DataSet, g *gbdt.GBDT, treecount int) (auc_score, logloss float32) {
	var totalweight float32 = 0.0
	auc := gbdt.NewAuc()
	for j := 0; j < len(testdataset.Samples); j++ {
		totalweight += testdataset.Samples[j].GetWeight()
		score := g.Predict(testdataset.Samples[j], treecount)
		p := gbdt.LogitCtr(score)
		auc.Add(float64(p), float64(testdataset.Samples[j].GetWeight()), testdataset.Samples[j].GetLabel())

		if testdataset.Samples[j].GetLabel() == 1 {
			if score < -15 {
				logloss -= score * 2 * testdataset.Samples[j].GetWeight()
			} else {
				logloss -= float32(math.Log(float64(p))) * testdataset.Samples[j].GetWeight()
			}
		} else {
			if score > 15 {
				logloss += score * 2 * testdataset.Samples[j].GetWeight()
			} else {
				logloss -= float32(math.Log(1-float64(p))) * testdataset.Samples[j].GetWeight()
			}
		}
	}
	auc_score = float32(auc.CalculateAuc())
	logloss = logloss / totalweight
	return
}
