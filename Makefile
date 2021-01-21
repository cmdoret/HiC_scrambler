READS1="./data/for.fq.gz"
READS2="./data/rev.fq.gz"
GENOME="./data/genome.fa"

.PHONY: clean demo test

clean:
	rm -rf demo_out demo_tmp

test:
	pytest

demo: clean
	python ./hic_scrambler/input_generator.py -1 $(READS1) -2 $(READS2) -t ./demo_tmp -n 2 $(GENOME) ./demo_out
