# 可取值：['densenet_lite_136']
ENCODER_NAME = densenet_lite_136
# 可取值：['fc', 'gru', 'lstm']
DECODER_NAME = fc
MODEL_NAME = $(ENCODER_NAME)-$(DECODER_NAME)
EPOCH = 41

INDEX_DIR = data/test
TRAIN_CONFIG_FP = docs/examples/train_config.json

# 训练模型
train:
	cnocr train -m $(MODEL_NAME) --index-dir $(INDEX_DIR) --train-config-fp $(TRAIN_CONFIG_FP)

# 在测试集上评估模型，所有badcases的具体信息会存放到文件夹 `evaluate/$(MODEL_NAME)` 中
evaluate:
	cnocr evaluate --model-name $(MODEL_NAME) -i data/test/dev.tsv \
		--image-folder data/images --batch-size 128 -o eval_results/$(MODEL_NAME)

predict:
	cnocr predict -m $(MODEL_NAME) -i docs/examples/rand_cn1.png

# build and serve mkdocs
doc:
#	pip install mkdocs
#	pip install mkdocs-macros-plugin
#	pip install mkdocs-material
#	pip install mkdocstrings
	python -m mkdocs serve
#	python -m mkdocs build


package:
	rm -rf build
	python setup.py sdist bdist_wheel

VERSION = 2.2.2.3
upload:
	python -m twine upload  dist/cnocr-$(VERSION)* --verbose

# 开启 OCR HTTP 服务
serve:
	cnocr serve -H 0.0.0.0 -p 8501 --reload

# 开启监控截屏文件夹的守护进程
daemon:
	python scripts/screenshot_daemon.py

docker-build:
	docker build -t breezedeus/cnocr:v$(VERSION) .

.PHONY: train evaluate predict doc package upload serve daemon
