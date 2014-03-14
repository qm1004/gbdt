package main

import (
	"fmt"
	"github.com/qm1004/gbdt"
	"log"
	"runtime"
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
	
}
