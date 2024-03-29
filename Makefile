READS1="./data/genome/for.fq.gz"
READS2="./data/genome/rev.fq.gz"
GENOME="./data/genome/genome.fa"

.PHONY: clean demo test deps setup

clean:
	rm -rf demo_out demo_tmp

test:
	pytest --doctest-modules

demo: clean
	python ./hic_scrambler/input_generator.py -1 $(READS1) -2 $(READS2) -t ./demo_tmp -n 10 $(GENOME) ./demo_out

train:
	python ./hic_scrambler/train.py
	
optim:
	python ./hic_scrambler/optim.py
	
predict:
	python ./hic_scrambler/predict.py

train_and_eval:
	python ./hic_scrambler/model.py
