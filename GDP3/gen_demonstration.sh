# use the same command as demonstration
# for example:
# bash GDP3/gen_demonstration.sh soccer 10




env_name=${1}
tar_num=${2}


python GDP3/gen_demonstration_expert.py --env_name=${env_name}  --num_episodes=${tar_num}
                           
                            



                                