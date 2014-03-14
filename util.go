package gbdt

import (
	"math/rand"
	"reflect"
	"time"
	"math"
)

func random_shuffle(array interface{}, l int) bool {
	if slice := reflect.ValueOf(array); slice.Kind() == reflect.Slice {
		if slice.Len() < l {
			return false
		}
		rand.Seed(time.Now().UTC().UnixNano())
		for i := 0; i < l; i++ {
			j := randInt(i, l)
			tmp := slice.Index(i).Interface()
			slice.Index(i).Set(slice.Index(j))
			slice.Index(j).Set(reflect.ValueOf(tmp))
		}
		return true
	}
	return false
}

func randInt(min int, max int) int {
	return min + rand.Intn(max-min)
}

func Float32Equal(a, b float32) bool {
	val := float32(math.Abs(float64(a - b)))
	if val < 1e-5 {
		return true
	}
	return false
}

func JoinString(vs []string, sep string) string {
	if len(vs) == 0 {
		return ""
	}
	res := vs[0]
	for i := 1; i < len(vs); i++ {
		res += sep
		res += vs[i]
	}
	return res
}