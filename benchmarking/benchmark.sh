#!/bin/bash

qsub -N benchmark -cwd  -l m_mem_free=100G -t 4-$(($(wc -l < results/benchmark_todo_2.csv) - 1))  -tc 100 job_runner.sh 
