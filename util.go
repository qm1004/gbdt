package gbdt

import (
	"math"
	"math/rand"
	"reflect"
	"sort"
	"strings"
	"time"
)

func random_shuffle(array interface{}, l int) bool {
	if slice := reflect.ValueOf(array); slice.Kind() == reflect.Slice {
		if slice.Len() < l {
			return false
		}
		length := slice.Len()
		rand.Seed(time.Now().UTC().UnixNano())
		for i := 0; i < l; i++ {
			j := RandInt(i, length)
			tmp := slice.Index(i).Interface()
			slice.Index(i).Set(slice.Index(j))
			slice.Index(j).Set(reflect.ValueOf(tmp))
		}
		return true
	}
	return false
}

func RandInt(min int, max int) int {
	return min + rand.Intn(max-min)
}

func Float32Equal(a, b float32) bool {
	val := float32(math.Abs(float64(a - b)))
	if val < 1e-5 {
		return true
	}
	return false
}

func Float32Square(a float32) float32 {
	return a * a
}

// A data structure to hold a key/value pair.
type Pair struct {
	Key   int
	Value float32
}

// A slice of Pairs that implements sort.Interface to sort by Value descend.
type PairList []Pair

func (p PairList) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }
func (p PairList) Len() int           { return len(p) }
func (p PairList) Less(i, j int) bool { return p[i].Value > p[j].Value }

func SortMapByValue(m map[int]float32) PairList {
	p := make(PairList, len(m))
	i := 0
	for k, v := range m {
		p[i] = Pair{k, v}
		i++
	}
	//fmt.Println("before:",p)
	sort.Sort(p)
	//fmt.Println("after:",p)
	return p
}

func LoadFeatureMap(s string) map[int]string {
	vs := strings.Split(s, "\n")
	feature_map := make(map[int]string)
	for i, val := range vs {
		feature_map[i] = val
	}
	return feature_map
}
