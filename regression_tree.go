package gbdt

import (
	"container/list"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"
)

var _ = os.Exit
var _ = fmt.Println
var _ = time.Now

const (
	LEFT          = 0
	RIGHT         = 1
	UNKNOWN       = 2
	CHILDSIZE     = 3
	UNKNOWN_VALUE = -math.MaxFloat32
)

type Node struct {
	child         []*Node
	feature_split Feature
	isleaf        bool
	pred          float32
	variance      float32
	sample_count  int //number of sample at this node
	depth         int
}

type RegressionTree struct {
	root          *Node
	max_depth     int
	min_leaf_size int
}

type NodeSample struct {
	node            *Node
	sample_sequence []int //sample sequence thie node included
}

type FeatureSplitInfo struct {
	feature_split Feature
	variance      float32
}

func NewRegressionTree() *RegressionTree {
	return &RegressionTree{
		root:          nil,
		max_depth:     Conf.Max_depth,
		min_leaf_size: Conf.Min_leaf_size,
	}
}

func (self *RegressionTree) Fit(d *DataSet, l int) {
	if l > len(d.samples) {
		log.Fatal("data length out of index")
	}

	self.root = &Node{
		child:        nil,
		isleaf:       false,
		pred:         0,
		variance:     0,
		sample_count: l,
		depth:        0,
	}

	//feature sampling
	featureid_list := make([]int, Conf.Number_of_feature)
	sampled_feature := make(map[int]bool)
	for i := 0; i < len(featureid_list); i++ {
		featureid_list[i] = i
		sampled_feature[i] = false
	}
	if Conf.Feature_sampling_ratio < 1 {
		random_shuffle(featureid_list, len(featureid_list)) //sample features for fitting tree
	}
	k := int(Conf.Feature_sampling_ratio * float32(Conf.Number_of_feature))
	for i := 0; i < k; i++ {
		sampled_feature[featureid_list[i]] = true
	}

	ns := &NodeSample{}
	ns.sample_sequence = make([]int, l)
	for i, _ := range d.samples {
		if i >= l {
			break
		}
		ns.sample_sequence[i] = i
	}
	ns.node = self.root

	queue := list.New()
	queue.PushBack(ns)

	sample_map_list := make([]*MapSample, l)
	for i, sample := range d.samples {
		if i >= l {
			break
		}
		sample_map := sample.ToMapSample()
		sample_map_list[i] = sample_map
	}
	predepth := 0
	for queue.Len() != 0 {
		var wg sync.WaitGroup
		predepth = queue.Front().Value.(*NodeSample).node.depth
		for queue.Len() != 0 {
			temp_ns := queue.Front()
			temp := temp_ns.Value.(*NodeSample)
			depth := temp.node.depth
			if predepth != depth {
				break
			}
			wg.Add(1)
			go func(d []*MapSample, node *Node, sample_sequence []int, queue *list.List, sampled_feature map[int]bool) {
				self.FitTree(d, node, sample_sequence, queue, sampled_feature)
				wg.Done()
			}(sample_map_list, temp.node, temp.sample_sequence, queue, sampled_feature)
			queue.Remove(temp_ns)

		}
		wg.Wait()

	}

}

func (self *RegressionTree) FitTree(d []*MapSample, node *Node, sample_sequence []int, queue *list.List, sampled_feature map[int]bool) {

	node.pred = NodePredictValue(d, sample_sequence)
	node.variance = CalculateVariance(d, sample_sequence)
	if node.depth >= self.max_depth || node.sample_count <= self.min_leaf_size || SameTarget(d, sample_sequence) {
		node.isleaf = true
		node.feature_split.id = -1
		return
	}

	if self.FindSplitFeature(d, node, sample_sequence, sampled_feature) == false {
		node.isleaf = true
		node.feature_split.id = -1
		return
	}
	child_sample_sequence := make([][]int, CHILDSIZE)
	index := node.feature_split.id
	split_value := node.feature_split.value
	for _, k := range sample_sequence {
		{
			if val, ok := d[k].feature[index]; !ok {
				child_sample_sequence[UNKNOWN] = append(child_sample_sequence[UNKNOWN], k)
			} else if ok {
				if val < split_value {
					child_sample_sequence[LEFT] = append(child_sample_sequence[LEFT], k)
				} else {
					child_sample_sequence[RIGHT] = append(child_sample_sequence[RIGHT], k)
				}
			}
		}
	}
	node.child = make([]*Node, CHILDSIZE)

	if len(child_sample_sequence[LEFT]) < self.min_leaf_size || len(child_sample_sequence[RIGHT]) < self.min_leaf_size {
		node.isleaf = true
		node.feature_split.id = -1
		return
	}
	node.child[LEFT] = &Node{child: nil, isleaf: false, pred: 0, variance: 0, sample_count: len(child_sample_sequence[LEFT]), depth: node.depth + 1}
	queue.PushBack(&NodeSample{node: node.child[LEFT], sample_sequence: child_sample_sequence[LEFT]})

	node.child[RIGHT] = &Node{child: nil, isleaf: false, pred: 0, variance: 0, sample_count: len(child_sample_sequence[RIGHT]), depth: node.depth + 1}
	queue.PushBack(&NodeSample{node: node.child[RIGHT], sample_sequence: child_sample_sequence[RIGHT]})

	if len(child_sample_sequence[UNKNOWN]) > self.min_leaf_size {
		node.child[UNKNOWN] = &Node{child: nil, isleaf: false, pred: 0, variance: 0, sample_count: len(child_sample_sequence[UNKNOWN]), depth: node.depth + 1}
		queue.PushBack(&NodeSample{node: node.child[UNKNOWN], sample_sequence: child_sample_sequence[UNKNOWN]})
	}

}

func (self *RegressionTree) FindSplitFeature(d []*MapSample, node *Node, sample_sequence []int, sampled_feature map[int]bool) bool {
	feature_tuple_list := make(map[int]*TupleList)

	for _, index := range sample_sequence { //build index for feature to samples
		known_valued_feature := make([]bool, Conf.Number_of_feature) //this sample has specific feature
		for fid, fvalue := range d[index].feature {
			if val, ok := sampled_feature[fid]; ok && val == true {

				if _, ok2 := feature_tuple_list[fid]; !ok2 {
					feature_tuple_list[fid] = NewTupleList()
				}
				feature_tuple_list[fid].AddTuple(fvalue, d[index].target, d[index].weight)
				known_valued_feature[fid] = true
			}
		}
		for fid, isknown := range known_valued_feature {
			if sampled_feature[fid] == true && isknown == false {
				feature_tuple_list[fid].AddTuple(UNKNOWN_VALUE, d[index].target, d[index].weight)
			}
		}
	}
	node.feature_split = Feature{id: -1, value: 0.0}
	var min_variance float32 = math.MaxFloat32

	var wg sync.WaitGroup
	chan_feature_split := make(chan *FeatureSplitInfo, len(feature_tuple_list)) //channel length
	for fid, t := range feature_tuple_list {                                    //find the best feature to split Node with
		wg.Add(1) //goroutine counter add 1
		go func(id int, tl *TupleList) {
			sort.Sort(tl)
			feature_split_info, ok := self.GetFeatureSplitValue(id, tl)
			if ok {
				chan_feature_split <- feature_split_info
			}
			wg.Done() //goroutine counter minus 1 when function return
		}(fid, t)
	}

	wg.Wait() //Wait blocks until the WaitGroup counter is zero
	close(chan_feature_split)
	for v := range chan_feature_split {
		if v != nil {
			temp_feature_split := v
			if min_variance > temp_feature_split.variance {
				min_variance = temp_feature_split.variance
				node.feature_split = temp_feature_split.feature_split
			}
		}

	}
	return min_variance != math.MaxFloat32
}

func (self *RegressionTree) GetFeatureSplitValue(fid int, t *TupleList) (*FeatureSplitInfo, bool) {
	var local_min_variance float32 = math.MaxFloat32
	var split_value float32 = 0.0
	var unknown int = 0

	var s, ss, total_weight float64 = 0.0, 0.0, 0.0
	var variance1, variance2, variance3 float32 = 0.0, 0.0, 0.0
	l := len(t.tuplelist)
	for unknown < l && t.tuplelist[unknown].value == UNKNOWN_VALUE { //calculate variance of unknown value samples for this feature
		s += float64(t.tuplelist[unknown].target * t.tuplelist[unknown].weight)
		ss += float64(t.tuplelist[unknown].target * t.tuplelist[unknown].target * t.tuplelist[unknown].weight)
		total_weight += float64(t.tuplelist[unknown].weight)
		unknown++
	}
	if unknown == l {
		return nil, false
	}
	if total_weight > 1 {
		variance1 = float32(ss - s*s/total_weight)
	} else {
		variance1 = 0
	}
	if variance1 < 0 {
		fmt.Println("variance1<0 for fid=", fid)
		variance1 = 0
	}
	s, ss, total_weight = 0, 0, 0
	for i := unknown; i < l; i++ {
		s += float64(t.tuplelist[i].target * t.tuplelist[i].weight)
		ss += float64(t.tuplelist[i].target * t.tuplelist[i].target * t.tuplelist[i].weight)
		total_weight += float64(t.tuplelist[i].weight)
	}

	var ls, lss, left_total_weight float64 = 0, 0, 0
	var rs, rss, right_total_weight float64 = s, ss, total_weight
	for i := unknown; i < l-1; i++ {
		s = float64(t.tuplelist[i].target * t.tuplelist[i].weight)
		ss = float64(t.tuplelist[i].target * t.tuplelist[i].target * t.tuplelist[i].weight)
		total_weight = float64(t.tuplelist[i].weight)

		ls += s
		lss += ss
		left_total_weight += total_weight

		rs -= s
		rss -= ss
		right_total_weight -= total_weight

		val1, val2 := t.tuplelist[i].value, t.tuplelist[i+1].value
		if Float32Equal(val1, val2) {
			continue
		}

		if left_total_weight > 1 {
			variance2 = float32(lss - ls*ls/left_total_weight)
		} else {
			variance2 = 0
		}
		if variance2 < 0 {
			//fmt.Println("variance2<0 for i=", i)
			variance2 = 0
		}

		if right_total_weight > 1 {
			variance3 = float32(rss - rs*rs/right_total_weight)
		} else {
			variance3 = 0
		}
		if variance3 < 0 {
			variance3 = 0
		}

		variance := variance1 + variance2 + variance3

		if local_min_variance > variance {
			local_min_variance = variance
			split_value = (val1 + val2) / 2.0
		}

	}
	feature_split_info := &FeatureSplitInfo{
		feature_split: Feature{id: fid, value: split_value},
		variance:      local_min_variance,
	}

	return feature_split_info, local_min_variance != math.MaxFloat32
}

func (self *RegressionTree) Predict(sample *Sample) float32 {
	node := self.root
	for {
		if node.isleaf {
			return node.pred
		}
		fid := node.feature_split.id
		split_value := node.feature_split.value
		if index, ok := sample.FindFeature(fid); ok == false {
			if node.child[UNKNOWN] != nil {
				node = node.child[UNKNOWN]
			} else {
				return node.pred
			}
		} else {
			if sample.feature[index].value < split_value {
				node = node.child[LEFT]
			} else {
				node = node.child[RIGHT]
			}
		}
	}
}

func (self *RegressionTree) Save() string {
	queue := list.New()
	position_map := make(map[*Node]int)
	self.SaveNodePos(self.root, queue, position_map)
	if queue.Len() == 0 {
		return ""
	}
	vs := make([]string, 0)
	for e := queue.Front(); e != nil; e = e.Next() {
		node := e.Value.(*Node)
		line := strconv.Itoa(position_map[node])
		line += "\t"
		line += strconv.Itoa(node.feature_split.id)
		line += "\t"
		line += strconv.FormatFloat(float64(node.feature_split.value), 'f', 4, 32)
		line += "\t"
		line += strconv.FormatBool(node.isleaf)
		line += "\t"
		line += strconv.FormatFloat(float64(node.pred), 'f', 4, 32)
		line += "\t"
		line += strconv.FormatFloat(float64(node.variance), 'f', 4, 32)
		line += "\t"
		line += strconv.Itoa(node.depth)
		line += "\t"
		line += strconv.Itoa(node.sample_count)
		for i := 0; i < CHILDSIZE; i++ {
			line += "\t"
			if node.child != nil && node.child[i] != nil {
				line += strconv.Itoa(position_map[node.child[i]])
			} else {
				line += "-1"
			}
		}
		vs = append(vs, line)
	}
	return strings.Join(vs, "\n")

}

func (self *RegressionTree) SaveNodePos(node *Node, queue *list.List, position_map map[*Node]int) {
	if node == nil {
		return
	}
	queue.PushBack(node)
	position_map[node] = queue.Len() - 1
	for e := queue.Front(); e != nil; e = e.Next() {
		temp_node := e.Value.(*Node)
		if temp_node != nil {
			for i := 0; i < CHILDSIZE; i++ {
				if temp_node.child != nil && temp_node.child[i] != nil {
					queue.PushBack(temp_node.child[i])
					position_map[temp_node.child[i]] = queue.Len() - 1

				}
			}

		}
	}
}

func (self *RegressionTree) Load(s string) {
	self.root = nil
	vs := strings.Split(s, "\n")
	left := make([]int, 0)
	right := make([]int, 0)
	unknown := make([]int, 0)
	nodes := make([]*Node, 0)
	for i := 0; i < len(vs); i++ {
		items := strings.Split(vs[i], "\t")
		node := &Node{}
		node.child = make([]*Node, CHILDSIZE)

		if id, err := strconv.ParseInt(items[1], 10, 0); err != nil {
			log.Fatal("feature_split.id", err)
		} else {
			node.feature_split.id = int(id)
		}

		if value, err := strconv.ParseFloat(items[2], 32); err != nil {
			log.Fatal("feature_split.value", err)
		} else {
			node.feature_split.value = float32(value)
		}

		if isleaf, err := strconv.ParseBool(items[3]); err != nil {
			log.Fatal("isleaf", err)
		} else {
			node.isleaf = isleaf
		}

		if pred, err := strconv.ParseFloat(items[4], 32); err != nil {
			log.Fatal("pred", err)
		} else {
			node.pred = float32(pred)
		}

		if variance, err := strconv.ParseFloat(items[5], 32); err != nil {
			log.Fatal("variance", err)
		} else {
			node.variance = float32(variance)
		}

		if depth, err := strconv.ParseInt(items[6], 10, 0); err != nil {
			log.Fatal("depth", err)
		} else {
			node.depth = int(depth)
		}
		if sample_count, err := strconv.ParseInt(items[7], 10, 0); err != nil {
			log.Fatal("sample_count", err)
		} else {
			node.sample_count = int(sample_count)
		}
		nodes = append(nodes, node)

		if lt, err := strconv.ParseInt(items[8], 10, 0); err != nil {
			log.Fatal("left", err)
		} else {
			left = append(left, int(lt))
		}
		if rt, err := strconv.ParseInt(items[9], 10, 0); err != nil {
			log.Fatal("right", err)
		} else {
			right = append(right, int(rt))
		}
		if un, err := strconv.ParseInt(items[10], 10, 0); err != nil {
			log.Fatal("unknown", err)
		} else {
			unknown = append(unknown, int(un))
		}
	}

	for i := 0; i < len(nodes); i++ {
		if left[i] >= 0 {
			nodes[i].child[LEFT] = nodes[left[i]]
		}
		if right[i] >= 0 {
			nodes[i].child[RIGHT] = nodes[right[i]]
		}
		if unknown[i] >= 0 {
			nodes[i].child[UNKNOWN] = nodes[unknown[i]]
		}
	}
	self.root = nodes[0]
}

func (self *RegressionTree) GetTreeFeatureWeight() map[int]float32 {
	feature_weight := make(map[int]float32)
	queue := list.New()
	if self.root != nil {
		queue.PushBack(self.root)
	}
	for e := queue.Front(); e != nil; e = e.Next() {
		node := e.Value.(*Node)
		if node.isleaf == false {
			queue.PushBack(node.child[LEFT])
			queue.PushBack(node.child[RIGHT])
			var gain float32 = 0.0
			if node.child[UNKNOWN] != nil {
				queue.PushBack(node.child[UNKNOWN])
				gain = node.variance - node.child[LEFT].variance - node.child[RIGHT].variance - node.child[UNKNOWN].variance
			} else {
				gain = node.variance - node.child[LEFT].variance - node.child[RIGHT].variance
			}
			if val, ok := feature_weight[node.feature_split.id]; ok {
				if gain > val {
					feature_weight[node.feature_split.id] = gain
				}
			} else {
				feature_weight[node.feature_split.id] = gain
			}
		} else {
			continue
		}
	}
	return feature_weight
}
