#!/bin/bash
echo "Sourcing ProxyR env"
source BP/anaconda3/bin/activate ProxyR
echo "Sourced ProxyR env"

experiment="x03"
startingepoch=1101
doepochs=200
endingepoch=1103 #$((startingepoch+doepochs))
echo -e "Training the VAE \n(=Running models/$experiment/autoencoder.py)\nStarting epoch = $startingepoch\nEnding epoch = $endingepoch"

for epoch in $(seq $startingepoch $endingepoch)
do
	echo "Epoch $epoch"
	python BP/Proxy_20CR/models/$experiment/autoencoder.py --epoch=$epoch
done

echo "Done!"
echo $(date '+%Y-%m-%d %H:%M:%S')


