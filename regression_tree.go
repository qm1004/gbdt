package gbdt

import (
	"container/list"
	"log"
	"math"
	"sort"
	"strconv"
	"sync"
	"strings"
)

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

func (self *RegressionTree) NewRegressionTree() *RegressionTree {
	return &RegressionTree{
		root:          nil,
		max_depth:     conf.max_depth,
		min_leaf_size: conf.min_leaf_size,
	}
}

func (self *RegressionTree) Fit(d *DataSet, l int) {
	if l > len(d) {
		log.Fatal("data length out of index")
	}

	self.root = &Node{
		child:        nil,
		isleaf:       false,
		pred:         0,
		variance:	  0,
		sample_count: l,
		depth:        0,
	}

	//feature sampling
	featureid_list := make([]int, conf.number_of_feature)
	sampled_feature := make(map[int]bool)
	for i := 0; i < len(featureid_list); i++ {
		featureid_list[i] = i
		sampled_feature[i] = false
	}
	if conf.feature_sampling_ratio < 1 {
		random_shuffle(featureid_list, len(featureid_list)) //sample features for fitting tree
	}
	k := int(conf.feature_sampling_ratio * conf.number_of_feature)
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
	ns.node = root

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

	for queue.Len() != 0 {
		temp_ns := queue.Front()
		queue.remove(temp_ns)
		temp := temp_ns.Value.(*NodeSample)
		self.Fit(sample_map_list, temp.node, temp.sample_sequence, queue, sampled_feature)
	}

}

func (self *RegressionTree) Fit(d []*MapSample, node *Node, sample_sequence []int, queue *list.List, sampled_feature map[int]bool) {

	node.pred = NodePredictValue(d, sample_sequence)
	node.variance = CalculateVariance(d,sample_sequence)
	if node.depth >= self.max_depth || node.sample_count <= self.min_leaf_size || SameTarget(d, node, sample_sequence) {
		node.isleaf = true
		return
	}

	if self.FindSplitFeature(d, node, sample_sequence, sampled_feature) == false {
		node.isleaf = true
		return
	}
	child_sample_sequence := make([][]int, CHILDSIZE)
	index := node.feature_split.id
	split_value := node.feature_split.value
	for i, k := range sample_sequence {
		{
			if val, ok := d[k].feature[index].value; !ok {
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
		return
	}
	node.child[LEFT] = &Node{child: nil, isleaf: false, pred: 0, variance:0,sample_count: len(child_sample_sequence[LEFT]), depth: node.depth + 1}
	queue.PushBack(&NodeSample{node: node.child[LEFT], sample_sequence: child_sample_sequence[LEFT]})

	node.child[RIGHT] = &Node{child: nil, isleaf: false, pred: 0,variance:0,sample_count: len(child_sample_sequence[RIGHT]), depth: node.depth + 1}
	queue.PushBack(&NodeSample{node: node.child[RIGHT], sample_sequence: child_sample_sequence[RIGHT]})

	if len(child_sample_sequence[UNKNOWN]) > self.min_leaf_size {
		node.child[UNKNOWN] = &Node{child: nil, isleaf: false, pred: 0,variance:0, sample_count: len(child_sample_sequence[UNKNOWN]), depth: node.depth + 1}
		queue.PushBack(&NodeSample{node: node.child[UNKNOWN], sample_sequence: child_sample_sequence[UNKNOWN]})
	}

}

func (self *RegressionTree) FindSplitFeature(d []*MapSample, node *Node, sample_sequence []int, sampled_feature map[int]bool) bool {
	feature_tuple_list := make(map[int]*TupleList)

	for _, index := range sample_sequence { //build index for feature to samples
		known_valued_feature := make([]bool, conf.number_of_feature) //this sample has specific feature
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
	chan_feature_split := make(chan *FeatureSplitInfo, 10) //channel length
	for fid, tuple_list := range feature_tuple_list {      //find the best feature to split Node with concurrency
		wg.Add(1) //goroutine counter add 1
		go func() {
			defer wg.Done() //goroutine counter minus 1 before function return
			sort.Sort(tuple_list)
			feature_split_info, ok := self.GetFeatureSplitValue(fid, tuple_list)
			if ok {
				chan_feature_split <- feature_split_info
			}
		}()
	}
	wg.Wait() //Wait blocks until the WaitGroup counter is zero
	close(chan_feature_split)
	for v := range chan_feature_split {
		if v != nil {
			temp_feature_split := v
			if min_variance > temp_feature_split.variance {
				min_variance = temp_feature_split.variance
				node.feature_split = temp_feature_split
			}
		}

	}
	return min_variance != math.MaxFloat32
}

func (self *RegressionTree) GetFeatureSplitValue(fid int, tuple_list *TupleList) (*FeatureSplitInfo, bool) {
	var local_min_variance float32 = math.MaxFloat32
	var split_value float32 = 0.0
	var unknown int = 0

	var s, ss, total_weight float64 = 0.0, 0.0, 0.0
	var variance1, variance2, variance3 float32 = 0.0, 0.0, 0.0
	l = len(tuple_list)
	for unknown < l && tuple_list[unknown].value == UNKNOWN_VALUE { //calculate variance of unknown value samples for this feature
		s += float64(tuple_list[unknown].target * tuple_list[unknown].weight)
		ss += float64(tuple_list[unknown].target * tuple_list[unknown].target * tuple_list[unknown].weight)
		total_weight += float64(tuple_list[unknown].weight)
		unknown++
	}
	if unknown == l {
		return nil, false
	}
	if total_weight > 1 {
		variance1 = float32(ss/total_weight - s*s/total_weight/total_weight)
	} else {
		variance1 = 0
	}
	if variance1 < 0 {
		log.Fatal("variance1<0!!")
		variance1 = 0
	}
	s, ss, total_weight = 0, 0, 0
	for i := unknown; i < l; i++ {
		s += float64(tuple_list[i].target * tuple_list[i].weight)
		ss += float64(tuple_list[i].target * tuple_list[i].target * tuple_list[i].weight)
		total_weight += float64(tuple_list[i].weight)
	}

	var ls, lss, left_total_weight float64 = 0, 0, 0
	var rs, rss, right_total_weight float64 = s, ss, total_weight
	for i := unknown; i < l-1; i++ {
		s = float64(tuple_list[i].target * tuple_list[i].weight)
		ss = float64(tuple_list[i].target * tuple_list[i].target * tuple_list[i].weight)
		total_weight = float64(tuple_list[i].weight)

		ls += s
		lss += ss
		left_total_weight += total_weight

		rs -= s
		rss -= ss
		right_total_weight -= total_weight

		val1, val2 := tuple_list[i].value, tuple_list[i+1].value
		if Float32Equal(val1, val2) {
			continue
		}

		if left_total_weight > 1 {
			variance2 = float32(lss/left_total_weight - ls*ls/left_total_weight/left_total_weight)
		} else {
			variance2 = 0
		}
		if variance2 < 0 {
			log.Fatal("variance2<0!!")
			variance2 = 0
		}

		if right_total_weight > 1 {
			variance3 = float32(rss/right_total_weight - rs*rs/right_total_weight/right_total_weight)
		} else {
			variance3 = 0
		}
		if variance3 < 0 {
			log.Fatal("variance3<0!!")
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
	map_sample := sample.ToMapSample()
	node := self.root
	for {
		if node.isleaf {
			return node.pred
		}
		fid := node.feature_split.id
		split_value := node.feature_split.value
		if val, ok := map_sample[fid]; !ok {
			if node.child[UNKNOWN] != nil {
				node = node.child[UNKNOWN]
			} else {
				return node.pred
			}
		} else {
			if val < split_value {
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
	self.SaveNodePos(self.root, queue, &position_map)
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
		line += strconv.FormatFloat(node.feature_split.value, "f", 32)
		line += "\t"
		line += strconv.FormatBool(node.isleaf)
		line += "\t"
		line += strconv.FormatFloat(node.pred, "f", 32)
		line += "\t"
		line += strconv.FormatFloat(node.variance, "f", 32)
		line += "\t"
		line += strconv.Itoa(node.depth)
		line += "\t"
		line += strconv.Itoa(node.sample_count)
		for i := 0; i < CHILDSIZE; i++ {
			line += "\t"
			if node.child[i]!=nil {
				line += strconv.Itoa(position_map[node])
			}else{
				line += "-1"
			}
		}
		vs=append(vs,line)
	}
	return strings.Join(vs,"\n")

}

func (self *RegressionTree) SaveNodePos(node *Node, queue *List, position_map *map[*node]int) {
	if !node {
		return
	}
	queue.PushBack(node)
	for e := queue.Front(); e != nil; e = e.Next() {
		temp_node := e.Value.(*Node)
		if temp_node != nil {
			position_map[temp_node] = queue.Len() - 1
			for i := 0; i < CHILDSIZE; i++ {
				if temp_node.child[i] != nil {
					queue.PushBack(temp_node.child[i])
				}
			}

		}
	}
}

func (self *RegressionTree) Load(s string) {
	self.root=nil
	vs:=strings.Split(s,"\n")
	left:=make([]int,0)
	right:=make([]int,0)
	unknown:=make([]int,0)
	nodes:=make([]*Node,0)
	for i := 0; i < len(vs); i++ {
		items:=strings.Split(vs[i],"\t")
		node:=&Node{}
		node.child:=make([]*Node,CHILDSIZE)
		var err error
		var lt,rt,un int
		if node.feature_split.id,err=strconv.ParseInt(items[1],10,0);err{
			log.Fatal("feature_split.id",err)
		}
		if node.feature_split.value,err=strconv.ParseFloat(items[2],32);err{
			log.Fatal("feature_split.value",err)
		}
		if node.isleaf,err=strconv.ParseBool(items[3]);err{
			log.Fatal("isleaf",err)
		}
		if node.pred,err=strconv.ParseFloat(items[4],32);err{
			log.Fatal("pred",err)
		}
		if node.variance,err=strconv.ParseFloat(items[5],32);err {
			log.Fatal("variance",err)
		}
		if node.depth,err=strconv.ParseInt(items[6],10,0);err{
			log.Fatal("depth",err)
		}
		if node.sample_count,err=strconv.ParseInt(items[7],10,0);err{
			log.Fatal("depth",err)
		}
		nodes=append(nodes,node)

		if lt,err=strconv.ParseInt(items[8],10,0);err{
			log.Fatal("left",err)
		}else{
			left=append(left,lt)
		}
		if rt,err=strconv.ParseInt(items[9],10,0);err{
			log.Fatal("right",err)
		}else{
			right=append(right,lt)
		}
		if un,err=strconv.ParseInt(items[10],10,0);err{
			log.Fatal("unknown",err)
		}else{
			unknown=append(unknown,un)
		}
	}

	for i := 0; i < len(nodes); i++ {
		if left[i]>=0 {
			nodes[i].child[LEFT]=nodes[left[i]]
		}
		if right[i]>=0 {
			nodes[i].child[RIGHT]=nodes[right[i]]
		}
		if unknown[i]>=0 {
			nodes[i].child[UNKNOWN]=nodes[unknown[i]]
		}
	}
	self.root=nodes[0]
}
