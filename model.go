package main

import (
	"fmt"
	"log"
	"runtime"

	torch "github.com/wangkuiyi/gotorch"
)

func main() {
	modelPath := "model/Adam_batch_size=2048_epoch=30.pt" // TorchScriptモデルのファイルパスを指定してください

	// GoTorchを初期化
	err := torch.InitJIT()
	if err != nil {
		log.Fatalf("Failed to initialize GoTorch: %v", err)
	}

	// TorchScriptモデルをロード
	jitModule, err := torch.JITLoad(modelPath)
	if err != nil {
		log.Fatalf("Failed to load TorchScript model: %v", err)
	}

	// モデルのパラメータを出力
	parameters := jitModule.Parameters()
	for _, param := range parameters {
		fmt.Printf("Parameter name: %s\n", param.Name())
		data := param.Data()
		fmt.Printf("Parameter data: %v\n", data)
	}

	// GoTorchの終了処理
	runtime.KeepAlive(parameters)
	torch.Finalize()
}
