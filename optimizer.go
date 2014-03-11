package gbdt

import (
	"log"
	"math"
)

const (
	LOG_LIKEHOOD = 1
	LEAST_SQUARE = 2
)

func FxGradient(y int, predict float32) float32 {
	switch conf.losstype {
	case LEAST_SQUARE:
		return y - predict
	case LOG_LIKEHOOD:
		return 2 * y / float32((1 + math.Exp(float64(2*float32(y)*predict))))
	}
	return 2 * y / float32((1 + math.Exp(float64(2*float32(y)*predict))))

}

func NodePredictValue(d []*MapSample, sample_sequence []int) float32 {
	switch conf.losstype {
	case LEAST_SQUARE:
		return LsOptimalValue(d, sample_sequence)
	case LOG_LIKEHOOD:
		return LogitOptimalValue(d, sample_sequence)
	}

	return LogitOptimalValue(d, sample_sequence)
}

func LogitOptimalValue(d []*MapSample, sample_sequence []int) float32 {
	var val1 float32 = 0
	var val2 float32 = 0
	if len(sample_sequence) > len(d) || len(sample_sequence) == 0 {
		return 0.0
	}
	for _, index := range sample_sequence {
		val1 += d[index].target
		absy := float32(math.Abs(float64(d[index].target)))
		val2 += absy / (2 - absy)
	}
	return val1 / val2
}

func LsOptimalValue(d []*MapSample, sample_sequence []int) float32 {
	var val1 float32 = 0
	var val2 float32 = 0
	if len(sample_sequence) > len(d) || len(sample_sequence) == 0 {
		return 0.0
	}
	for _, index := range sample_sequence {
		val1 += d[index].target * d[index].weight
		val2 += d[index].weight
	}
	return val1 / val2
}

func SameTarget(d []*MapSample, sample_sequence []int) bool {
	if len(sample_sequence) == 0 || len(sample_sequence) < len(d) {
		log.Fatal("out of index sample_sequence")
	}
	var val float32 = d[sample_sequence[0]].target
	for i := 1; i < len(sample_sequence); i++ {
		if !Float32Equal(val, d[sample_sequence[i]].target) {
			return false
		}
	}
	return true
}

func CalculateVariance(d []*MapSample, sample_sequence []int) (variance float32){
	var s,ss,total_weight float32=0.0,0.0,0.0
	for _,k:=range sample_sequence {
		s += float64(MapSample[k].target * MapSample[k].weight)
		ss += float64(MapSample[k].target * MapSample[k].target * MapSample[k].weight)
		total_weight += float64(MapSample[k].weight)
	}
	if total_weight > 1 {
		variance = float32(ss/total_weight - s*s/total_weight/total_weight)
	} else {
		variance = 0
	}
}

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
