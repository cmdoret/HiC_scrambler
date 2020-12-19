
.PHONY: clean demo test

clean:
	rm -rf demo_out demo_tmp

test:
	pytest

demo: clean
	python ./hic_scrambler/input_generator.py -1 ./data/for.fq.gz -2 ./data/rev.fq.gz -t ./demo_tmp -n 2 ./data/genome.fa ./demo_out
