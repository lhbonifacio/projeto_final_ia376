# IA376: Projeto Final

Este repositório contém a implementação referente ao projeto final da disciplina IA376 do segundo semestre de 2020 ofertada pela FEEC-UNICAMP.
Nesta projeto estão disponíveis as implementações para o pré-treinamento de um modelo baseado na combinação dos modelos EfficientNet e T5, para a tarefa de extração de informações de recibos.

Com base neste código, é possível realizar o pré-treinamento do modelo proposto, utilizando dois datasets distintos, além de realizar a avaliação da tarefa final, utilizando o dataset alvo.

##Datasets
Após clonar o repositório, é necessário fazer o download dos datasets utilizados:
###Synthetic Word Dataset
```shell
    $ wget https://www.robots.ox.ac.uk/~vgg/data/text/mjsynth.tar.gz
```

###SROIE ICDAR-2019
```shell
    $ gsutil -m cp -n gs://neuralresearcher_data/unicamp/ia376j_2020s2/aula8/dataset_sroie_icdar_2019.zip .
```
Para executar o pré-treinamento/treinamento:
```shell
$ python main.py --dataset sroie --train_file $path/to/train --val_file $path/to/validation --test_file $path/to/test \
--model_type $t5-model-type --output_dir $path/to/output --batch_size 16 --num_workers 4
```