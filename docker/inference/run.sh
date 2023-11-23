#!/bin/bash

#help
display_help()
{
   echo "HViT inference, by @clementgrisi"
   echo
   echo "Syntax: docker run hvit [-f filepath] [-o folder] [-k wandb_api_key]"
   echo "options:"
   echo "-f     input csv file"
   echo "-o     output folder"
   echo "-k     (optional) wandb API key"
   echo
}

#main
while getopts ":t:f:o:h" opt; do
  case $opt in
    h)
      display_help
      exit 1
      ;;
    f)
      csv="$OPTARG"
      ;;
    o)
      output="$OPTARG"
      ;;
    k)
      key="$OPTARG"
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

if [[ -n $key ]]; then
  export WANDB_API_KEY=$key
fi

#hs2p
/usr/local/bin/python3 hs2p/patch_extraction.py --config-name "panda_patch_extraction" slide_csv="${csv}"

#hvit
## feature extraction ##
folds=(0 1 2 3 4)
for i in ${!folds[@]}
do
  fold=${folds[i]}
	/usr/local/bin/python3 -m torch.distributed.run --standalone --nproc_per_node=gpu hvit/extract_features.py --config-name "panda_feature_extraction" fold=${fold}
done
## model inference ##
/usr/local/bin/python3 hvit/inference/ensemble.py --config-name "panda_inference" test_csv="${csv}"

cp hvit/output/inference/results/submission.csv ${output}/submission.csv