#PEMSD3-stream
# python main.py --conf config/PEMSD3-stream/model.json --dataset PEMSD3-stream --gpuid 1
# python main.py --conf config/PEMSD3-stream/retrained.json --dataset PEMSD3-stream --gpuid 1
# python main.py --conf config/PEMSD3-stream/static.json --dataset PEMSD3-stream --gpuid 1

# python main.py --conf config/PEMSD3-stream/no_TMRB.json --dataset PEMSD3-stream --gpuid 1
# python main.py --conf config/PEMSD3-stream/no_update.json --dataset PEMSD3-stream --gpuid 1
# python main.py --conf config/PEMSD3-stream/no_select.json --dataset PEMSD3-stream --gpuid 1

# python main.py --conf config/PEMSD3-stream/no_replay.json --dataset PEMSD3-stream --gpuid 1
# python main.py --conf config/PEMSD3-stream/no_increase.json --dataset PEMSD3-stream --gpuid 1

#PEMSD4-large
# python main.py --conf config/PEMSD4-large/model.json --dataset PEMSD4-stream --gpuid 1
# python main.py --conf config/PEMSD4-large/retrained.json --dataset PEMSD4-stream --gpuid 1
# python main.py --conf config/PEMSD4-large/static.json --dataset PEMSD4-stream --gpuid 1


# python main.py --conf config/PEMSD4-large/no_TMRB.json --dataset PEMSD4-large --gpuid 1
# python main.py --conf config/PEMSD4-large/no_update.json --dataset PEMSD4-large --gpuid 1
# python main.py --conf config/PEMSD4-large/no_select.json --dataset PEMSD4-large --gpuid 1

# python main.py --conf config/PEMSD4-large/no_replay.json --dataset PEMSD4-large --gpuid 1
# python main.py --conf config/PEMSD4-large/no_increase.json --dataset PEMSD4-large --gpuid 1

#PEMSD8-mini
python main.py --conf config/PEMSD8-mini/model.json --dataset PEMSD8-mini --gpuid 1
# python main.py --conf config/PEMSD8-mini/retrained.json --dataset PEMSD8-mini --gpuid 1
# python main.py --conf config/PEMSD8-mini/static.json --dataset PEMSD8-mini --gpuid 1

# python main.py --conf config/PEMSD8-mini/no_TMRB.json --dataset PEMSD8-mini --gpuid 1
# python main.py --conf config/PEMSD8-mini/no_update.json --dataset PEMSD8-mini --gpuid 1
# python main.py --conf config/PEMSD8-mini/no_select.json --dataset PEMSD8-mini --gpuid 1

# python main.py --conf config/PEMSD8-mini/no_replay.json --dataset PEMSD8-mini --gpuid 1
# python main.py --conf config/PEMSD8-mini/no_increase.json --dataset PEMSD8-mini --gpuid 1


#parameter
#topk
# python main.py --conf config/parameter/topk/PEMSD3/topk16.json --dataset PEMSD3-stream --gpuid 1
# python main.py --conf config/parameter/topk/PEMSD3/topk8.json --dataset PEMSD3-stream --gpuid 1
# python main.py --conf config/parameter/topk/PEMSD3/topk4.json --dataset PEMSD3-stream --gpuid 1
# python main.py --conf config/parameter/topk/PEMSD3/topk20.json --dataset PEMSD3-stream --gpuid 1

# python main.py --conf config/parameter/topk/PEMSD4/topk16.json --dataset PEMSD4-large --gpuid 1
# python main.py --conf config/parameter/topk/PEMSD4/topk8.json --dataset PEMSD4-large --gpuid 1
# python main.py --conf config/parameter/topk/PEMSD4/topk4.json --dataset PEMSD4-large --gpuid 1
# python main.py --conf config/parameter/topk/PEMSD4/topk20.json --dataset PEMSD4-large --gpuid 1

# python main.py --conf config/parameter/topk/PEMSD8/topk16.json --dataset PEMSD8-mini --gpuid 1
# python main.py --conf config/parameter/topk/PEMSD8/topk8.json --dataset PEMSD8-mini --gpuid 1
# python main.py --conf config/parameter/topk/PEMSD8/topk4.json --dataset PEMSD8-mini --gpuid 1
# python main.py --conf config/parameter/topk/PEMSD8/topk20.json --dataset PEMSD8-mini --gpuid 1

#rho
# python main.py --conf config/parameter/rho/PEMSD3/rho0.json --dataset PEMSD3-stream --gpuid 1
# python main.py --conf config/parameter/rho/PEMSD3/rho0.03.json --dataset PEMSD3-stream --gpuid 1
# python main.py --conf config/parameter/rho/PEMSD3/rho0.08.json --dataset PEMSD3-stream --gpuid 1
# python main.py --conf config/parameter/rho/PEMSD3/rho0.1.json --dataset PEMSD3-stream --gpuid 1
# python main.py --conf config/parameter/rho/PEMSD3/rho0.15.json --dataset PEMSD3-stream --gpuid 1

# python main.py --conf config/parameter/rho/PEMSD4/rho0.json --dataset PEMSD4-large --gpuid 1
# python main.py --conf config/parameter/rho/PEMSD4/rho0.05.json --dataset PEMSD4-large --gpuid 1
# python main.py --conf config/parameter/rho/PEMSD4/rho0.08.json --dataset PEMSD4-large --gpuid 1
# python main.py --conf config/parameter/rho/PEMSD4/rho0.1.json --dataset PEMSD4-large --gpuid 1
# python main.py --conf config/parameter/rho/PEMSD4/rho0.15.json --dataset PEMSD4-large --gpuid 1

# python main.py --conf config/parameter/rho/PEMSD8/rho0.json --dataset PEMSD8-mini --gpuid 1
# python main.py --conf config/parameter/rho/PEMSD8/rho0.1.json --dataset PEMSD8-mini --gpuid 1
# python main.py --conf config/parameter/rho/PEMSD8/rho0.15.json --dataset PEMSD8-mini --gpuid 1
# python main.py --conf config/parameter/rho/PEMSD8/rho0.2.json --dataset PEMSD8-mini --gpuid 1
# python main.py --conf config/parameter/rho/PEMSD8/rho0.3.json --dataset PEMSD8-mini --gpuid 1