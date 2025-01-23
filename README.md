# Adventures in North American *Sphagnum* section *Sphagnum* computer vision

## DATASETS

TensorFlow Record (.tfr) files containing images in PNG format images (species identification numbers are in alphabetical order):

* [5-species dataset](https://drive.google.com/file/d/1ffUw4jktsy8jA1X4LZJ1UYuKoxI1aU7n/view?usp=sharing) [315MB]
* [12-species dataset](https://drive.google.com/file/d/1EAzCxz0eIN9m0NVy7D-kjztssycC0iQa/view?usp=sharing) [611MB]

## CODE

Code developed and tested using TensorFlow 2.13.0 [official Docker images](https://hub.docker.com/r/tensorflow/tensorflow/tags).

<!--

COUNTS="{0:4995,1:37,2:422,3:3248,4:108,5:762,6:8310,7:6057,8:12057,9:876,10:169,11:35}"
DIR=FireNetSEz
GPU=0
MODEL=x.keras
SEED=42
SPECIES=12
TEST=Sphagnum-Sphagnum-vegetative64-test.tfr
TRAIN=Sphagnum-Sphagnum-12ua-vegetative64-train.tfr
VAL_ALL=Sphagnum-Sphagnum-12ua-vegetative64-validation.tfr
VAL_ONE=Sphagnum-Sphagnum-vegetative64-validation.tfr

-->

* Create a FireNetSEz model: 
```bash
./firenet.py -a $SPECIES -c -f selu -i 64 -o $MODEL -p 0 -r $SEED -s -x 48,32,32,48 -z conv ### 12-species = 100,320 parameters
```
* Pretrain the model:
```bash
./trainImagesC.py -a $SPECIES -b 16 -C "[0.8,1.4,0.0]" -d 0.01 -e 4096 -f ccm+clr+aw -g $GPU -i 64 -l 0.001 -m $MODEL -o $DIR -Q -r $SEED -t $TRAIN -v $VAL_ALL
```
* Evaluate the pretraining: 
```bash
LAST=$(ls -ltr $DIR/*/best-model.keras | awk -F"/" "{print \$2}" | tail -1)
./testImages.py -a $SPECIES -b 96 -g $GPU -i 64 -j -m $DIR"/"$LAST"/best-model.keras" -t $VAL_ONE
```
* Make a greedy model soup:
```bash
LAST=$(ls -ltr $DIR/*/best-model.keras | awk -F"/" "{print \$2}" | tail -1 | perl -pe "s/-best/-intermediate/")
./imageSoup.py -a $SPECIES -b 96 -d $TRAIN -g $GPU -m $DIR"/"$LAST -o $DIR -T 60 -v $VAL_ALL
```
* Evaluate the model soup: 
```bash
LAST=$(ls -ltr $DIR/*/soup-model.keras | awk -F"/" "{print \$2}" | tail -1)
./testImages.py -a $SPECIES -b 96 -g $GPU -i 64 -j -m $DIR"/"$LAST"/soup-model.keras" -t $VAL_ONE
```
* Finetune the model soup: 
```bash
./trainImagesC.py -a $SPECIES -b 16 -d 0.0001 -e 4096 -f ce+clr+aw -g $GPU -i 64 -k 0.4 -l 0.00001 -m $DIR"/"$LAST"/soup-model.keras" -o $DIR -q -r $SEED -s "output_gap" -t $TRAIN -u 0.4 -v $VAL_ALL -y "$COUNTS" 
```
* Evaluate the finetuned model:
```bash
LAST=$(ls -ltr $DIR/*/best-model.keras | awk -F"/" "{print \$2}" | tail -1)
./testImages.py -a 12 -b 96 -g $GPU -i $SIZE -j -m $DIR"/"$LAST"/best-model.keras" -t $TEST
```
