# Adventures in North American *Sphagnum* subgenus *Sphagnum* computer vision

## DATASETS

TensorFlow Record (.tfr) files containing images in PNG format and best models:
* 10-species dataset [[476M]](https://drive.google.com/file/d/1sE3qGkcK9CH80hi0wlNvQ70PsgxGgdPg/view?usp=sharing)
* 5-species dataset [[220M]](https://drive.google.com/file/d/1ECb0uo64iT0Bd7s_pdLuz4Azf3queDaF/view?usp=sharing)
* 5-species verified [[197M]](https://drive.google.com/file/d/1faPBSYIuamF5E5lTrVtsvZc1ueFynC05/view?usp=sharing)

| *SPHAGNUM* SUBGENUS *SPHAGNUM*       | 10-SPECIES INDEX | 5-SPECIES INDEX |
| ------------------------------------ | ---------------- | --------------- |
| *S. affine* Renauld & Cardot         | 0                | 0               |
| *S. alaskense* R.E.Andrus & Janssens | 1                | NULL            |
| *S. austinii* Sull. in C.F.Austin    | 2                | NULL            |
| *S. centrale* C.E.O.Jensen           | 3                | 1               |
| *S. magellanicum* Brid.              | 4                | 2               |
| *S. palustre* L.                     | 5                | 3               |
| *S. papillosum* Lindb.               | 6                | 4               |
| *S. perichaetiale* Hampe             | 7                | NULL            |
| *S. portoricense* Hampe              | 8                | NULL            |
| *S. steerei* R.E.Andrus              | 9                | NULL            |

## CODE

Code developed and tested using TensorFlow 2.13.0 [official Docker images](https://hub.docker.com/r/tensorflow/tensorflow/tags).

<!--

COUNTS="{0:4995,1:37,2:422,3:3248,4:8310,5:6057,6:12057,7:876,8:169,9:35}"
DIR=FireNetSEz
GPU=0
MODEL=x.keras
SEED=42
SIZE=64
SPECIES=10
TEST=Sphagnum-Sphagnum-10b-vegetative-test.tfr
TRAIN=Sphagnum-Sphagnum-10ua-vegetative64-train.tfr
VALIDATION=Sphagnum-Sphagnum-10ua-vegetative64-validation.tfr

-->

* Create a FireNetSEz model: 
```bash
SIZE=64
./firenet.py -a $SPECIES -c -f selu -i $SIZE -o $MODEL -p 0 -r $SEED -s -x 48,32,32,48 -Z 5 -z conv ### 10-species = 100,126 parameters; 5-species = 99,641 parameters

```
* Pretrain the model:
```bash
GPU=0
./trainImagesC.py -a $SPECIES -b 16 -C "[0.8,1.4,0.0]" -d 0.01 -e 4096 -f ccm+clr+aw -g $GPU -i $SIZE -l 0.001 -m $MODEL -o $DIR -Q -r $SEED -t $TRAIN -v $VALIDATION
```
* Evaluate the pretraining: 
```bash
LAST=$(ls -ltr $DIR/*/best-model.keras | awk -F"/" "{print \$2}" | tail -1)
./testImagesPNG.py -a $SPECIES -b 96 -g $GPU -i $SIZE -m $DIR"/"$LAST"/best-model.keras" -t $VALIDATION
```
* Make a greedy model soup:
```bash
LAST=$(ls -ltr $DIR/*/best-model.keras | awk -F"/" "{print \$2}" | tail -1 | perl -pe "s/-best/-intermediate/")
./imageSoup.py -a $SPECIES -b 96 -d $TRAIN -g $GPU -m $DIR"/"$LAST -o $DIR -T 60 -v $VALIDATION
```
* Evaluate the model soup: 
```bash
LAST=$(ls -ltr $DIR/*/soup-model.keras | awk -F"/" "{print \$2}" | tail -1)
./testImagesPNG.py -a $SPECIES -b 96 -g $GPU -i $SIZE -m $DIR"/"$LAST"/soup-model.keras" -t $VALIDATION
```
* Finetune the model soup: 
```bash
./trainImagesC.py -a $SPECIES -b 16 -d 0.0001 -e 4096 -f ce+clr+aw -g $GPU -i $SIZE -k 0.4 -l 0.00001 -m $DIR"/"$LAST"/soup-model.keras" -o $DIR -q -r $SEED -s "output_gap" -t $TRAIN -u 0.4 -v $VALIDATION -y "$COUNTS" 
```
* Evaluate the finetuned model:
```bash
LAST=$(ls -ltr $DIR/*/best-model.keras | awk -F"/" "{print \$2}" | tail -1)
./testImagesMeanPNG.py -a $SPECIES -g $GPU -i $SIZE -m $DIR"/"$LAST"/best-model.keras" -t $TEST
```

## CITATION

If you use the *Sphagnum* subgenus *Sphagnum* datasets, code, or models in your work, please [cite](https://doi.org/10.1111/nph.70461) us.
